import os
import struct
import unittest
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from nibabel import nifti1 as nifti1
from nibabel.affines import from_matvec
from nibabel.casting import have_binary128, type_info
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import (
from nibabel.optpkg import optional_package
from nibabel.pkg_info import cmp_pkg_version
from nibabel.spatialimages import HeaderDataError
from nibabel.tmpdirs import InTemporaryDirectory
from ..freesurfer import load as mghload
from ..orientations import aff2axcodes
from ..testing import (
from . import test_analyze as tana
from . import test_spm99analyze as tspm
from .nibabel_data import get_nibabel_data, needs_nibabel_data
from .test_arraywriters import IUINT_TYPES, rt_err_estimate
from .test_orientations import ALL_ORNTS
class TestNifti1PairHeader(tana.TestAnalyzeHeader, tspm.HeaderScalingMixin):
    header_class = Nifti1PairHeader
    example_file = header_file
    quat_dtype = np.float32
    supported_np_types = tana.TestAnalyzeHeader.supported_np_types.union((np.int8, np.uint16, np.uint32, np.int64, np.uint64, np.complex128))
    if have_binary128():
        supported_np_types = supported_np_types.union((np.longdouble, np.clongdouble))
    tana.add_duplicate_types(supported_np_types)

    def test_empty(self):
        tana.TestAnalyzeHeader.test_empty(self)
        hdr = self.header_class()
        assert hdr['magic'] == hdr.pair_magic
        assert hdr['scl_slope'] == 1
        assert hdr['vox_offset'] == 0

    def test_from_eg_file(self):
        hdr = self.header_class.from_fileobj(open(self.example_file, 'rb'))
        assert hdr.endianness == '<'
        assert hdr['magic'] == hdr.pair_magic
        assert hdr['sizeof_hdr'] == self.sizeof_hdr

    def test_data_scaling(self):
        super().test_data_scaling()
        hdr = self.header_class()
        data = np.arange(0, 3, 0.5).reshape((1, 2, 3))
        hdr.set_data_shape(data.shape)
        hdr.set_data_dtype(np.float32)
        S = BytesIO()
        hdr.data_to_fileobj(data, S, rescale=True)
        assert_array_equal(hdr.get_slope_inter(), (1, 0))
        rdata = hdr.data_from_fileobj(S)
        assert_array_almost_equal(data, rdata)
        hdr.set_data_dtype(np.int8)
        hdr.set_slope_inter(1, 0)
        hdr.data_to_fileobj(data, S, rescale=True)
        assert not np.allclose(hdr.get_slope_inter(), (1, 0))
        rdata = hdr.data_from_fileobj(S)
        assert_array_almost_equal(data, rdata)
        hdr.set_slope_inter(1, 0)
        with np.errstate(invalid='ignore'):
            hdr.data_to_fileobj(data, S, rescale=False)
        assert_array_equal(hdr.get_slope_inter(), (1, 0))
        rdata = hdr.data_from_fileobj(S)
        assert_array_almost_equal(np.round(data), rdata)

    def test_big_scaling(self):
        hdr = self.header_class()
        hdr.set_data_shape((2, 1, 1))
        hdr.set_data_dtype(np.int16)
        sio = BytesIO()
        dtt = np.float32
        finf = type_info(dtt)
        data = np.array([finf['min'], finf['max']], dtype=dtt)[:, None, None]
        hdr.data_to_fileobj(data, sio)
        data_back = hdr.data_from_fileobj(sio)
        assert np.allclose(data, data_back)

    def test_slope_inter(self):
        hdr = self.header_class()
        nan, inf, minf = (np.nan, np.inf, -np.inf)
        HDE = HeaderDataError
        assert hdr.get_slope_inter() == (1.0, 0.0)
        for in_tup, exp_err, out_tup, raw_values in (((None, None), None, (None, None), (nan, nan)), ((nan, None), None, (None, None), (nan, nan)), ((None, nan), None, (None, None), (nan, nan)), ((nan, nan), None, (None, None), (nan, nan)), ((None, 0), HDE, (None, None), (nan, 0)), ((nan, 0), HDE, (None, None), (nan, 0)), ((1, None), HDE, (None, None), (1, nan)), ((1, nan), HDE, (None, None), (1, nan)), ((0, 0), HDE, (None, None), (0, 0)), ((0, None), HDE, (None, None), (0, nan)), ((0, nan), HDE, (None, None), (0, nan)), ((0, inf), HDE, (None, None), (0, inf)), ((0, minf), HDE, (None, None), (0, minf)), ((inf, 0), HDE, (None, None), (inf, 0)), ((inf, None), HDE, (None, None), (inf, nan)), ((inf, nan), HDE, (None, None), (inf, nan)), ((inf, inf), HDE, (None, None), (inf, inf)), ((inf, minf), HDE, (None, None), (inf, minf)), ((minf, 0), HDE, (None, None), (minf, 0)), ((minf, None), HDE, (None, None), (minf, nan)), ((minf, nan), HDE, (None, None), (minf, nan)), ((minf, inf), HDE, (None, None), (minf, inf)), ((minf, minf), HDE, (None, None), (minf, minf)), ((2, None), HDE, HDE, (2, nan)), ((2, nan), HDE, HDE, (2, nan)), ((2, inf), HDE, HDE, (2, inf)), ((2, minf), HDE, HDE, (2, minf)), ((2, 0), None, (2, 0), (2, 0)), ((2, 1), None, (2, 1), (2, 1))):
            hdr = self.header_class()
            if not exp_err is None:
                with pytest.raises(exp_err):
                    hdr.set_slope_inter(*in_tup)
                in_list = [v if not v is None else np.nan for v in in_tup]
                hdr['scl_slope'], hdr['scl_inter'] = in_list
            else:
                hdr.set_slope_inter(*in_tup)
                if isinstance(out_tup, Exception):
                    with pytest.raises(out_tup):
                        hdr.get_slope_inter()
                else:
                    assert hdr.get_slope_inter() == out_tup
                    hdr = self.header_class.from_header(hdr, check=True)
                    assert hdr.get_slope_inter() == out_tup
            assert_array_equal([hdr['scl_slope'], hdr['scl_inter']], raw_values)

    def test_nifti_qfac_checks(self):
        hdr = self.header_class()
        hdr['pixdim'][0] = 1
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = -1
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = 0
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert fhdr['pixdim'][0] == 1
        assert message == 'pixdim[0] (qfac) should be 1 (default) or -1; setting qfac to 1'

    def test_nifti_qsform_checks(self):
        HC = self.header_class
        hdr = HC()
        hdr['qform_code'] = -1
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert fhdr['qform_code'] == 0
        assert message == 'qform_code -1 not valid; setting to 0'
        hdr = HC()
        hdr['sform_code'] = -1
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert fhdr['sform_code'] == 0
        assert message == 'sform_code -1 not valid; setting to 0'

    def test_nifti_xform_codes(self):
        hdr = self.header_class()
        affine = np.eye(4)
        for code in nifti1.xform_codes.keys():
            hdr.set_qform(affine, code)
            assert hdr['qform_code'] == nifti1.xform_codes[code]
            hdr.set_sform(affine, code)
            assert hdr['sform_code'] == nifti1.xform_codes[code]
        for bad_code in (-1, 6, 10):
            with pytest.raises(KeyError):
                hdr.set_qform(affine, bad_code)
            with pytest.raises(KeyError):
                hdr.set_sform(affine, bad_code)

    def test_magic_offset_checks(self):
        HC = self.header_class
        hdr = HC()
        hdr['magic'] = 'ooh'
        fhdr, message, raiser = self.log_chk(hdr, 45)
        assert fhdr['magic'] == b'ooh'
        assert message == "magic string 'ooh' is not valid; leaving as is, but future errors are likely"
        svo = hdr.single_vox_offset
        for magic, ok, bad_spm in ((hdr.pair_magic, 32, 40), (hdr.single_magic, svo + 32, svo + 40)):
            hdr['magic'] = magic
            hdr['vox_offset'] = 0
            self.assert_no_log_err(hdr)
            hdr['vox_offset'] = ok
            self.assert_no_log_err(hdr)
            hdr['vox_offset'] = bad_spm
            fhdr, message, raiser = self.log_chk(hdr, 30)
            assert fhdr['vox_offset'] == bad_spm
            assert message == f'vox offset (={bad_spm:g}) not divisible by 16, not SPM compatible; leaving at current value'
        hdr['magic'] = hdr.single_magic
        hdr['vox_offset'] = 10
        fhdr, message, raiser = self.log_chk(hdr, 40)
        assert fhdr['vox_offset'] == hdr.single_vox_offset
        assert message == 'vox offset 10 too low for single file nifti1; setting to minimum value of ' + str(hdr.single_vox_offset)

    def test_freesurfer_large_vector_hack(self):
        HC = self.header_class
        hdr = HC()
        hdr.set_data_shape((2, 3, 4))
        assert hdr.get_data_shape() == (2, 3, 4)
        assert hdr['glmin'] == 0
        dim_type = hdr.template_dtype['dim'].base
        glmin = hdr.template_dtype['glmin'].base
        too_big = int(np.iinfo(dim_type).max) + 1
        hdr.set_data_shape((too_big - 1, 1, 1))
        assert hdr.get_data_shape() == (too_big - 1, 1, 1)
        full_shape = (too_big, 1, 1, 1, 1, 1, 1)
        for dim in range(3, 8):
            expected_dim = np.array([dim, -1, 1, 1, 1, 1, 1, 1])
            with suppress_warnings():
                hdr.set_data_shape(full_shape[:dim])
            assert hdr.get_data_shape() == full_shape[:dim]
            assert_array_equal(hdr['dim'], expected_dim)
            assert hdr['glmin'] == too_big
        with suppress_warnings():
            hdr.set_data_shape((too_big, 1, 1, 4))
        assert hdr.get_data_shape() == (too_big, 1, 1, 4)
        assert_array_equal(hdr['dim'][:5], np.array([4, -1, 1, 1, 4]))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (too_big,))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (too_big, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (too_big, 1, 2))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (too_big, 2, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, too_big))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, too_big, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 1, too_big))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 1, 1, too_big))
        far_too_big = int(np.iinfo(glmin).max) + 1
        with suppress_warnings():
            hdr.set_data_shape((far_too_big - 1, 1, 1))
        assert hdr.get_data_shape() == (far_too_big - 1, 1, 1)
        with pytest.raises(HeaderDataError):
            hdr.set_data_shape((far_too_big, 1, 1))
        hdr.set_data_shape((-1, 1, 1))
        hdr['glmin'] = 0
        with pytest.raises(HeaderDataError):
            hdr.get_data_shape()
        for shape in ((too_big - 1, 1, 1), (too_big, 1, 1)):
            for constructor in (list, tuple, np.array):
                with suppress_warnings():
                    hdr.set_data_shape(constructor(shape))
                assert hdr.get_data_shape() == shape

    @needs_nibabel_data('nitest-freesurfer')
    def test_freesurfer_ico7_hack(self):
        HC = self.header_class
        hdr = HC()
        full_shape = (163842, 1, 1, 1, 1, 1, 1)
        for dim in range(3, 8):
            expected_dim = np.array([dim, 27307, 1, 6, 1, 1, 1, 1])
            hdr.set_data_shape(full_shape[:dim])
            assert hdr.get_data_shape() == full_shape[:dim]
            assert_array_equal(hdr._structarr['dim'], expected_dim)
        pytest.raises(HeaderDataError, hdr.set_data_shape, full_shape[:1])
        pytest.raises(HeaderDataError, hdr.set_data_shape, full_shape[:2])
        pytest.raises(HeaderDataError, hdr.set_data_shape, (163842, 2, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (163842, 1, 2))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 163842, 1))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 1, 163842))
        pytest.raises(HeaderDataError, hdr.set_data_shape, (1, 1, 1, 163842))
        nitest_path = os.path.join(get_nibabel_data(), 'nitest-freesurfer')
        mgh = mghload(os.path.join(nitest_path, 'fsaverage', 'surf', 'lh.orig.avg.area.mgh'))
        nii = load(os.path.join(nitest_path, 'derivative', 'fsaverage', 'surf', 'lh.orig.avg.area.nii'))
        assert mgh.shape == nii.shape
        assert_array_equal(mgh.get_fdata(), nii.get_fdata())
        assert_array_equal(nii.header._structarr['dim'][1:4], np.array([27307, 1, 6]))
        with InTemporaryDirectory():
            nii.to_filename('test.nii')
            nii2 = load('test.nii')
            assert nii.shape == nii2.shape
            assert_array_equal(nii.get_fdata(), nii2.get_fdata())
            assert_array_equal(nii.affine, nii2.affine)

    def test_qform_sform(self):
        HC = self.header_class
        hdr = HC()
        assert_array_equal(hdr.get_qform(), np.eye(4))
        empty_sform = np.zeros((4, 4))
        empty_sform[-1, -1] = 1
        assert_array_equal(hdr.get_sform(), empty_sform)
        assert hdr.get_qform(coded=True) == (None, 0)
        assert hdr.get_sform(coded=True) == (None, 0)
        nice_aff = np.diag([2, 3, 4, 1])
        another_aff = np.diag([3, 4, 5, 1])
        nasty_aff = from_matvec(np.arange(9).reshape((3, 3)), [9, 10, 11])
        nasty_aff[0, 0] = 1
        fixed_aff = unshear_44(nasty_aff)
        assert not np.allclose(fixed_aff, nasty_aff)
        for in_meth, out_meth in ((hdr.set_qform, hdr.get_qform), (hdr.set_sform, hdr.get_sform)):
            in_meth(nice_aff, 2)
            aff, code = out_meth(coded=True)
            assert_array_equal(aff, nice_aff)
            assert code == 2
            assert_array_equal(out_meth(), nice_aff)
            in_meth(another_aff, 0)
            assert out_meth(coded=True) == (None, 0)
            assert_array_almost_equal(out_meth(), another_aff)
            in_meth(nice_aff)
            aff, code = out_meth(coded=True)
            assert code == 2
            in_meth(nice_aff, 1)
            in_meth(nice_aff)
            aff, code = out_meth(coded=True)
            assert code == 1
            assert_array_equal(aff, nice_aff)
            in_meth(None, 3)
            aff, code = out_meth(coded=True)
            assert_array_equal(aff, nice_aff)
            assert code == 3
            in_meth(None, 0)
            assert out_meth(coded=True) == (None, 0)
            in_meth(None)
            assert out_meth(coded=True) == (None, 0)
            in_meth(nice_aff.tolist())
            assert_array_equal(out_meth(), nice_aff)
        hdr.set_qform(nasty_aff, 1)
        assert_array_almost_equal(hdr.get_qform(), fixed_aff)
        with pytest.raises(HeaderDataError):
            hdr.set_qform(nasty_aff, 1, False)
        hdr.set_sform(None)
        hdr.set_qform(nice_aff, 1)
        assert hdr.get_sform(coded=True) == (None, 0)
        hdr.set_sform(nasty_aff, 1)
        aff, code = hdr.get_sform(coded=True)
        assert_array_equal(aff, nasty_aff)
        assert code == 1

    def test_datatypes(self):
        hdr = self.header_class()
        for code in data_type_codes.value_set():
            dt = data_type_codes.type[code]
            if dt == np.void:
                continue
            hdr.set_data_dtype(code)
            assert hdr.get_data_dtype() == data_type_codes.dtype[code]
        hdr.set_data_dtype(np.complex128)
        hdr.check_fix()

    def test_quaternion(self):
        hdr = self.header_class()
        hdr['quatern_b'] = 0
        hdr['quatern_c'] = 0
        hdr['quatern_d'] = 0
        assert np.allclose(hdr.get_qform_quaternion(), [1.0, 0, 0, 0])
        hdr['quatern_b'] = 1
        hdr['quatern_c'] = 0
        hdr['quatern_d'] = 0
        assert np.allclose(hdr.get_qform_quaternion(), [0, 1, 0, 0])
        hdr['quatern_b'] = 1 + np.finfo(self.quat_dtype).eps
        assert_array_almost_equal(hdr.get_qform_quaternion(), [0, 1, 0, 0])

    def test_qform(self):
        ehdr = self.header_class()
        ehdr.set_qform(A)
        qA = ehdr.get_qform()
        assert np.allclose(A, qA, atol=1e-05)
        assert np.allclose(Z, ehdr['pixdim'][1:4])
        xfas = nifti1.xform_codes
        assert ehdr['qform_code'] == xfas['aligned']
        ehdr.set_qform(A, 'scanner')
        assert ehdr['qform_code'] == xfas['scanner']
        ehdr.set_qform(A, xfas['aligned'])
        assert ehdr['qform_code'] == xfas['aligned']
        for dims in ((-1, 1, 1), (1, -1, 1), (1, 1, -1)):
            ehdr['pixdim'][1:4] = dims
            with pytest.raises(HeaderDataError):
                ehdr.get_qform()

    def test_sform(self):
        ehdr = self.header_class()
        ehdr.set_sform(A)
        sA = ehdr.get_sform()
        assert np.allclose(A, sA, atol=1e-05)
        xfas = nifti1.xform_codes
        assert ehdr['sform_code'] == xfas['aligned']
        ehdr.set_sform(A, 'scanner')
        assert ehdr['sform_code'] == xfas['scanner']
        ehdr.set_sform(A, xfas['aligned'])
        assert ehdr['sform_code'] == xfas['aligned']

    def test_dim_info(self):
        ehdr = self.header_class()
        assert ehdr.get_dim_info() == (None, None, None)
        for info in ((0, 2, 1), (None, None, None), (0, 2, None), (0, None, None), (None, 2, 1), (None, None, 1)):
            ehdr.set_dim_info(*info)
            assert ehdr.get_dim_info() == info

    def test_slice_times(self):
        hdr = self.header_class()
        with pytest.raises(HeaderDataError):
            hdr.get_slice_times()
        hdr.set_dim_info(slice=2)
        with pytest.raises(HeaderDataError):
            hdr.get_slice_times()
        hdr.set_data_shape((1, 1, 7))
        with pytest.raises(HeaderDataError):
            hdr.get_slice_times()
        hdr.set_slice_duration(0.1)
        _stringer = lambda val: val is not None and '%2.1f' % val or None
        _print_me = lambda s: list(map(_stringer, s))
        hdr['slice_code'] = slice_order_codes['sequential increasing']
        assert _print_me(hdr.get_slice_times()) == ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']
        hdr['slice_start'] = 1
        hdr['slice_end'] = 5
        assert _print_me(hdr.get_slice_times()) == [None, '0.0', '0.1', '0.2', '0.3', '0.4', None]
        hdr['slice_code'] = slice_order_codes['sequential decreasing']
        assert _print_me(hdr.get_slice_times()) == [None, '0.4', '0.3', '0.2', '0.1', '0.0', None]
        hdr['slice_code'] = slice_order_codes['alternating increasing']
        assert _print_me(hdr.get_slice_times()) == [None, '0.0', '0.3', '0.1', '0.4', '0.2', None]
        hdr['slice_code'] = slice_order_codes['alternating decreasing']
        assert _print_me(hdr.get_slice_times()) == [None, '0.2', '0.4', '0.1', '0.3', '0.0', None]
        hdr['slice_code'] = slice_order_codes['alternating increasing 2']
        assert _print_me(hdr.get_slice_times()) == [None, '0.2', '0.0', '0.3', '0.1', '0.4', None]
        hdr['slice_code'] = slice_order_codes['alternating decreasing 2']
        assert _print_me(hdr.get_slice_times()) == [None, '0.4', '0.1', '0.3', '0.0', '0.2', None]
        hdr = self.header_class()
        hdr.set_dim_info(slice=2)
        times = [None, 0.2, 0.4, 0.1, 0.3, 0.0, None]
        with pytest.raises(HeaderDataError):
            hdr.set_slice_times(times)
        hdr.set_data_shape([1, 1, 7])
        with pytest.raises(HeaderDataError):
            hdr.set_slice_times(times[:-1])
        with pytest.raises(HeaderDataError):
            hdr.set_slice_times((None,) * len(times))
        n_mid_times = times[:]
        n_mid_times[3] = None
        with pytest.raises(HeaderDataError):
            hdr.set_slice_times(n_mid_times)
        funny_times = times[:]
        funny_times[3] = 0.05
        with pytest.raises(HeaderDataError):
            hdr.set_slice_times(funny_times)
        hdr.set_slice_times(times)
        assert hdr.get_value_label('slice_code') == 'alternating decreasing'
        assert hdr['slice_start'] == 1
        assert hdr['slice_end'] == 5
        assert_array_almost_equal(hdr['slice_duration'], 0.1)
        hdr2 = self.header_class()
        hdr2.set_dim_info(slice=2)
        hdr2.set_slice_duration(0.1)
        hdr2.set_data_shape((1, 1, 2))
        with pytest.warns(UserWarning) as w:
            hdr2.set_slice_times([0.1, 0])
            assert len(w) == 1
        assert hdr2.get_value_label('slice_code') == 'sequential decreasing'
        with pytest.warns(UserWarning) as w:
            hdr2.set_slice_times([0, 0.1])
            assert len(w) == 1
        assert hdr2.get_value_label('slice_code') == 'sequential increasing'

    def test_intents(self):
        ehdr = self.header_class()
        ehdr.set_intent('t test', (10,), name='some score')
        assert ehdr.get_intent() == ('t test', (10.0,), 'some score')
        with pytest.raises(KeyError):
            ehdr.set_intent('no intention')
        with pytest.raises(KeyError):
            ehdr.set_intent('no intention', allow_unknown=True)
        with pytest.raises(KeyError):
            ehdr.set_intent(32767)
        with pytest.raises(HeaderDataError):
            ehdr.set_intent('t test', (10, 10))
        with pytest.raises(HeaderDataError):
            ehdr.set_intent('f test', (10,))
        ehdr.set_intent('t test')
        assert (ehdr['intent_p1'], ehdr['intent_p2'], ehdr['intent_p3']) == (0, 0, 0)
        assert ehdr['intent_name'] == b''
        ehdr.set_intent('t test', (10,))
        assert (ehdr['intent_p2'], ehdr['intent_p3']) == (0, 0)
        ehdr.set_intent(9999, allow_unknown=True)
        assert ehdr.get_intent() == ('unknown code 9999', (), '')
        assert ehdr.get_intent('code') == (9999, (), '')
        ehdr.set_intent(9999, name='custom intent', allow_unknown=True)
        assert ehdr.get_intent() == ('unknown code 9999', (), 'custom intent')
        assert ehdr.get_intent('code') == (9999, (), 'custom intent')
        ehdr.set_intent(code=9999, params=(1, 2, 3), allow_unknown=True)
        assert ehdr.get_intent() == ('unknown code 9999', (), '')
        assert ehdr.get_intent('code') == (9999, (), '')
        with pytest.raises(HeaderDataError):
            ehdr.set_intent(999, (1,), allow_unknown=True)
        with pytest.raises(HeaderDataError):
            ehdr.set_intent(999, (1, 2), allow_unknown=True)

    def test_set_slice_times(self):
        hdr = self.header_class()
        hdr.set_dim_info(slice=2)
        hdr.set_data_shape([1, 1, 7])
        hdr.set_slice_duration(0.1)
        times = [0] * 6
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None] * 7
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None, 0, 1, None, 3, 4, None]
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None, 0, 1, 2.1, 3, 4, None]
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None, 0, 4, 3, 2, 1, None]
        pytest.raises(HeaderDataError, hdr.set_slice_times, times)
        times = [0, 1, 2, 3, 4, 5, 6]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 1
        assert hdr['slice_start'] == 0
        assert hdr['slice_end'] == 6
        assert hdr['slice_duration'] == 1.0
        times = [None, 0, 1, 2, 3, 4, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 1
        assert hdr['slice_start'] == 1
        assert hdr['slice_end'] == 5
        assert hdr['slice_duration'] == 1.0
        times = [None, 0.4, 0.3, 0.2, 0.1, 0, None]
        hdr.set_slice_times(times)
        assert np.allclose(hdr['slice_duration'], 0.1)
        times = [None, 4, 3, 2, 1, 0, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 2
        times = [None, 0, 3, 1, 4, 2, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 3
        times = [None, 2, 4, 1, 3, 0, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 4
        times = [None, 2, 0, 3, 1, 4, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 5
        times = [None, 4, 1, 3, 0, 2, None]
        hdr.set_slice_times(times)
        assert hdr['slice_code'] == 6

    def test_xyzt_units(self):
        hdr = self.header_class()
        assert hdr.get_xyzt_units() == ('unknown', 'unknown')
        hdr.set_xyzt_units('mm', 'sec')
        assert hdr.get_xyzt_units() == ('mm', 'sec')
        hdr.set_xyzt_units()
        assert hdr.get_xyzt_units() == ('unknown', 'unknown')

    def test_recoded_fields(self):
        hdr = self.header_class()
        assert hdr.get_value_label('qform_code') == 'unknown'
        hdr['qform_code'] = 3
        assert hdr.get_value_label('qform_code') == 'talairach'
        assert hdr.get_value_label('sform_code') == 'unknown'
        hdr['sform_code'] = 3
        assert hdr.get_value_label('sform_code') == 'talairach'
        assert hdr.get_value_label('intent_code') == 'none'
        hdr.set_intent('t test', (10,), name='some score')
        assert hdr.get_value_label('intent_code') == 't test'
        assert hdr.get_value_label('slice_code') == 'unknown'
        hdr['slice_code'] = 4
        assert hdr.get_value_label('slice_code') == 'alternating decreasing'