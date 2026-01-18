import gzip
from copy import copy
from decimal import Decimal
from hashlib import sha1
from os.path import dirname
from os.path import join as pjoin
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...volumeutils import endian_codes
from .. import dicomreaders as didr
from .. import dicomwrappers as didw
from . import dicom_test, have_dicom, pydicom
class TestMultiFrameWrapper(TestCase):
    MINIMAL_MF = {'PerFrameFunctionalGroupsSequence': [None], 'SharedFunctionalGroupsSequence': [None]}
    WRAPCLASS = didw.MultiframeWrapper

    @dicom_test
    def test_shape(self):
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        with pytest.raises(didw.WrapperError):
            dw.image_shape
        fake_mf['Rows'] = 64
        with pytest.raises(didw.WrapperError):
            dw.image_shape
        fake_mf.pop('Rows')
        fake_mf['Columns'] = 64
        with pytest.raises(didw.WrapperError):
            dw.image_shape
        fake_mf['Rows'] = 32
        with pytest.raises(AssertionError):
            dw.image_shape
        fake_mf['NumberOfFrames'] = 4
        with pytest.raises(AssertionError):
            dw.image_shape
        div_seq = ((1, 1), (1, 2), (1, 3), (1, 4))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        assert MFW(fake_mf).image_shape == (32, 64, 4)
        div_seq = ((1, 1), (1, 2), (1, 3), (2, 4))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        div_seq = ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 2, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 3)
        div_seq = ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (2, 2, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        div_seq = ((1, 1, 1), (1, 2, 1), (1, 1, 3), (1, 2, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 2)
        div_seq = ((1, 1, 0), (1, 2, 0), (1, 1, 3), (1, 2, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 2)
        div_seq = ((1,), (2,), (3,), (4,))
        sid_seq = (1, 1, 1, 1)
        fake_mf.update(fake_shape_dependents(div_seq, sid_seq=sid_seq))
        assert MFW(fake_mf).image_shape == (32, 64, 4)
        div_seq = ((1,), (2,), (3,), (4,))
        sid_seq = (1, 1, 1, 2)
        fake_mf.update(fake_shape_dependents(div_seq, sid_seq=sid_seq))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        div_seq = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3))
        sid_seq = (1, 1, 1, 1, 1, 1)
        fake_mf.update(fake_shape_dependents(div_seq, sid_seq=sid_seq))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 3)
        div_seq = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3))
        sid_seq = (1, 1, 1, 1, 1, 2)
        fake_mf.update(fake_shape_dependents(div_seq, sid_seq=sid_seq))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        div_seq = ((1, 1), (2, 1), (3, 1), (4, 1))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=1))
        assert MFW(fake_mf).image_shape == (32, 64, 4)
        div_seq = ((1, 1), (2, 1), (3, 2), (4, 1))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=1))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        div_seq = ((1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (1, 1, 3), (2, 1, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=1))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 3)

    def test_iop(self):
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        with pytest.raises(didw.WrapperError):
            dw.image_orient_patient
        fake_frame = fake_frames('PlaneOrientationSequence', 'ImageOrientationPatient', [[0, 1, 0, 1, 0, 0]])[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_orient_patient, [[0, 1], [1, 0], [0, 0]])
        fake_mf['SharedFunctionalGroupsSequence'] = [None]
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_orient_patient
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_orient_patient, [[0, 1], [1, 0], [0, 0]])

    def test_voxel_sizes(self):
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        with pytest.raises(didw.WrapperError):
            dw.voxel_sizes
        fake_frame = fake_frames('PixelMeasuresSequence', 'PixelSpacing', [[2.1, 3.2]])[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).voxel_sizes
        fake_mf['SpacingBetweenSlices'] = 4.3
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 4.3])
        fake_frame.PixelMeasuresSequence[0].SliceThickness = 5.4
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
        del fake_mf['SpacingBetweenSlices']
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
        fake_mf['SharedFunctionalGroupsSequence'] = [None]
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).voxel_sizes
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
        fake_frame = fake_frames('PixelMeasuresSequence', 'PixelSpacing', [[Decimal('2.1'), Decimal('3.2')]])[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        fake_mf['SpacingBetweenSlices'] = Decimal('4.3')
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 4.3])
        fake_frame.PixelMeasuresSequence[0].SliceThickness = Decimal('5.4')
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])

    def test_image_position(self):
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        with pytest.raises(didw.WrapperError):
            dw.image_position
        fake_frame = fake_frames('PlanePositionSequence', 'ImagePositionPatient', [[-2.0, 3.0, 7]])[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_position, [-2, 3, 7])
        fake_mf['SharedFunctionalGroupsSequence'] = [None]
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_position
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_position, [-2, 3, 7])
        fake_frame.PlanePositionSequence[0].ImagePositionPatient = [Decimal(str(v)) for v in [-2, 3, 7]]
        assert_array_equal(MFW(fake_mf).image_position, [-2, 3, 7])
        assert MFW(fake_mf).image_position.dtype == float

    @dicom_test
    @pytest.mark.xfail(reason='Not packaged in install', raises=FileNotFoundError)
    def test_affine(self):
        dw = didw.wrapper_from_file(DATA_FILE_4D)
        aff = dw.affine

    @dicom_test
    @pytest.mark.xfail(reason='Not packaged in install', raises=FileNotFoundError)
    def test_data_real(self):
        dw = didw.wrapper_from_file(DATA_FILE_4D)
        data = dw.get_data()
        if endian_codes[data.dtype.byteorder] == '>':
            data = data.byteswap()
        dat_str = data.tobytes()
        assert sha1(dat_str).hexdigest() == '149323269b0af92baa7508e19ca315240f77fa8c'

    @dicom_test
    def test_slicethickness_fallback(self):
        dw = didw.wrapper_from_file(DATA_FILE_EMPTY_ST)
        assert dw.voxel_sizes[2] == 1.0

    @dicom_test
    @needs_nibabel_data('nitest-dicom')
    def test_data_derived_shape(self):
        dw = didw.wrapper_from_file(DATA_FILE_4D_DERIVED)
        with pytest.warns(UserWarning, match='Derived images found and removed'):
            assert dw.image_shape == (96, 96, 60, 33)

    @dicom_test
    @needs_nibabel_data('dcm_qa_xa30')
    def test_data_trace(self):
        dw = didw.wrapper_from_file(DATA_FILE_SIEMENS_TRACE)
        assert dw.image_shape == (72, 72, 39, 1)

    @dicom_test
    @needs_nibabel_data('nitest-dicom')
    def test_data_unreadable_private_headers(self):
        with pytest.warns(UserWarning, match='Error while attempting to read CSA header'):
            dw = didw.wrapper_from_file(DATA_FILE_CT)
        assert dw.image_shape == (512, 571)

    @dicom_test
    def test_data_fake(self):
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        with pytest.raises(didw.WrapperError):
            dw.get_data()
        dw.image_shape = (2, 3, 4)
        with pytest.raises(didw.WrapperError):
            dw.get_data()
        fake_mf['Rows'] = 2
        fake_mf['Columns'] = 3
        dim_idxs = ((1, 1), (1, 2), (1, 3), (1, 4))
        fake_mf.update(fake_shape_dependents(dim_idxs, sid_dim=0))
        assert MFW(fake_mf).image_shape == (2, 3, 4)
        with pytest.raises(didw.WrapperError):
            dw.get_data()
        data = np.arange(24).reshape((2, 3, 4))
        fake_mf['pixel_array'] = np.rollaxis(data, 2)
        dw = MFW(fake_mf)
        assert_array_equal(dw.get_data(), data)
        fake_mf['RescaleSlope'] = 2.0
        fake_mf['RescaleIntercept'] = -1
        assert_array_equal(MFW(fake_mf).get_data(), data * 2.0 - 1)
        dim_idxs = ((1, 4), (1, 2), (1, 3), (1, 1))
        fake_mf.update(fake_shape_dependents(dim_idxs, sid_dim=0))
        sorted_data = data[..., [3, 1, 2, 0]]
        fake_mf['pixel_array'] = np.rollaxis(sorted_data, 2)
        assert_array_equal(MFW(fake_mf).get_data(), data * 2.0 - 1)
        dim_idxs = [[1, 4, 2, 1], [1, 2, 2, 1], [1, 3, 2, 1], [1, 1, 2, 1], [1, 4, 2, 2], [1, 2, 2, 2], [1, 3, 2, 2], [1, 1, 2, 2], [1, 4, 1, 1], [1, 2, 1, 1], [1, 3, 1, 1], [1, 1, 1, 1], [1, 4, 1, 2], [1, 2, 1, 2], [1, 3, 1, 2], [1, 1, 1, 2]]
        fake_mf.update(fake_shape_dependents(dim_idxs, sid_dim=0))
        shape = (2, 3, 4, 2, 2)
        data = np.arange(np.prod(shape)).reshape(shape)
        sorted_data = data.reshape(shape[:2] + (-1,), order='F')
        order = [11, 9, 10, 8, 3, 1, 2, 0, 15, 13, 14, 12, 7, 5, 6, 4]
        sorted_data = sorted_data[..., np.argsort(order)]
        fake_mf['pixel_array'] = np.rollaxis(sorted_data, 2)
        assert_array_equal(MFW(fake_mf).get_data(), data * 2.0 - 1)

    def test__scale_data(self):
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        data = np.arange(24).reshape((2, 3, 4))
        assert_array_equal(data, dw._scale_data(data))
        fake_mf['RescaleSlope'] = 2.0
        fake_mf['RescaleIntercept'] = -1.0
        assert_array_equal(data * 2 - 1, dw._scale_data(data))
        fake_frame = fake_frames('PixelValueTransformationSequence', 'RescaleSlope', [3.0])[0]
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        dw = MFW(fake_mf)
        with pytest.raises(AttributeError):
            dw._scale_data(data)
        fake_frame.PixelValueTransformationSequence[0].RescaleIntercept = -2
        assert_array_equal(data * 3 - 2, dw._scale_data(data))
        fake_frame.PixelValueTransformationSequence[0].RescaleSlope = Decimal('3')
        fake_frame.PixelValueTransformationSequence[0].RescaleIntercept = Decimal('-2')
        assert_array_equal(data * 3 - 2, dw._scale_data(data))