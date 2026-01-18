import copy
import os
import sys
import unittest
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arr_dict_equal, clear_and_catch_warnings, data_path, error_warnings
from .. import trk as trk_module
from ..header import Field
from ..tractogram import Tractogram
from ..tractogram_file import HeaderError, HeaderWarning
from ..trk import (
from .test_tractogram import assert_tractogram_equal
class TestTRK(unittest.TestCase):

    def test_load_empty_file(self):
        for lazy_load in [False, True]:
            trk = TrkFile.load(DATA['empty_trk_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(trk.tractogram, DATA['empty_tractogram'])

    def test_load_simple_file(self):
        for lazy_load in [False, True]:
            trk = TrkFile.load(DATA['simple_trk_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(trk.tractogram, DATA['simple_tractogram'])

    def test_load_complex_file(self):
        for lazy_load in [False, True]:
            trk = TrkFile.load(DATA['complex_trk_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(trk.tractogram, DATA['complex_tractogram'])

    def trk_with_bytes(self, trk_key='simple_trk_fname', endian='<'):
        """Return example trk file bytes and struct view onto bytes"""
        with open(DATA[trk_key], 'rb') as fobj:
            trk_bytes = bytearray(fobj.read())
        dt = trk_module.header_2_dtype.newbyteorder(endian)
        trk_struct = np.ndarray((1,), dt, buffer=trk_bytes)
        trk_struct.flags.writeable = True
        return (trk_struct, trk_bytes)

    def test_load_file_with_wrong_information(self):
        trk_struct1, trk_bytes1 = self.trk_with_bytes()
        trk_struct1[Field.VOXEL_ORDER] = b'LAS'
        trk1 = TrkFile.load(BytesIO(trk_bytes1))
        trk_struct2, trk_bytes2 = self.trk_with_bytes()
        trk_struct2[Field.VOXEL_ORDER] = b'las'
        trk2 = TrkFile.load(BytesIO(trk_bytes2))
        trk1_aff2rasmm = get_affine_trackvis_to_rasmm(trk1.header)
        trk2_aff2rasmm = get_affine_trackvis_to_rasmm(trk2.header)
        assert_array_equal(trk1_aff2rasmm, trk2_aff2rasmm)
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.NB_STREAMLINES] = 0
        trk = TrkFile.load(BytesIO(trk_bytes), lazy_load=False)
        assert_tractogram_equal(trk.tractogram, DATA['simple_tractogram'])
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.VOXEL_TO_RASMM] = np.zeros((4, 4))
        with pytest.warns(HeaderWarning, match='identity'):
            trk = TrkFile.load(BytesIO(trk_bytes))
        assert_array_equal(trk.affine, np.eye(4))
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.VOXEL_TO_RASMM] = np.diag([0, 0, 0, 1])
        with clear_and_catch_warnings(record=True, modules=[trk_module]) as w:
            with pytest.raises(HeaderError):
                TrkFile.load(BytesIO(trk_bytes))
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.VOXEL_ORDER] = b''
        with pytest.warns(HeaderWarning, match='LPS'):
            TrkFile.load(BytesIO(trk_bytes))
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct['version'] = 123
        with pytest.raises(HeaderError):
            TrkFile.load(BytesIO(trk_bytes))
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct['hdr_size'] = 1234
        with pytest.raises(HeaderError):
            TrkFile.load(BytesIO(trk_bytes))
        trk_struct, trk_bytes = self.trk_with_bytes('complex_trk_fname')
        trk_struct['scalar_name'][0, 0] = b'colors\x003\x004'
        with pytest.raises(HeaderError):
            TrkFile.load(BytesIO(trk_bytes))
        trk_struct, trk_bytes = self.trk_with_bytes('complex_trk_fname')
        trk_struct['property_name'][0, 0] = b'colors\x003\x004'
        with pytest.raises(HeaderError):
            TrkFile.load(BytesIO(trk_bytes))

    def test_load_trk_version_1(self):
        trk_struct, trk_bytes = self.trk_with_bytes()
        trk_struct[Field.VOXEL_TO_RASMM] = np.diag([2, 3, 4, 1])
        trk = TrkFile.load(BytesIO(trk_bytes))
        assert_array_equal(trk.affine, np.diag([2, 3, 4, 1]))
        trk_struct['version'] = 1
        with pytest.warns(HeaderWarning, match='identity'):
            trk = TrkFile.load(BytesIO(trk_bytes))
        assert_array_equal(trk.affine, np.eye(4))
        assert_array_equal(trk.header['version'], 1)

    def test_load_complex_file_in_big_endian(self):
        trk_struct, trk_bytes = self.trk_with_bytes('complex_trk_big_endian_fname', endian='>')
        good_orders = '>' if sys.byteorder == 'little' else '>='
        hdr_size = trk_struct['hdr_size']
        assert hdr_size.dtype.byteorder in good_orders
        assert hdr_size == 1000
        for lazy_load in [False, True]:
            trk = TrkFile.load(DATA['complex_trk_big_endian_fname'], lazy_load=lazy_load)
            with pytest.warns(Warning) if lazy_load else error_warnings():
                assert_tractogram_equal(trk.tractogram, DATA['complex_tractogram'])

    def test_tractogram_file_properties(self):
        trk = TrkFile.load(DATA['simple_trk_fname'])
        assert trk.streamlines == trk.tractogram.streamlines
        assert_array_equal(trk.affine, trk.header[Field.VOXEL_TO_RASMM])

    def test_write_empty_file(self):
        tractogram = Tractogram(affine_to_rasmm=np.eye(4))
        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file)
        assert_tractogram_equal(new_trk.tractogram, tractogram)
        new_trk_orig = TrkFile.load(DATA['empty_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)
        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['empty_trk_fname'], 'rb').read()

    def test_write_simple_file(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file)
        assert_tractogram_equal(new_trk.tractogram, tractogram)
        new_trk_orig = TrkFile.load(DATA['simple_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)
        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['simple_trk_fname'], 'rb').read()

    def test_write_complex_file(self):
        tractogram = Tractogram(DATA['streamlines'], data_per_point=DATA['data_per_point'], affine_to_rasmm=np.eye(4))
        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)
        data_per_streamline = DATA['data_per_streamline']
        tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tractogram)
        trk_file = BytesIO()
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)
        data_per_streamline = DATA['data_per_streamline']
        tractogram = Tractogram(DATA['streamlines'], data_per_point=DATA['data_per_point'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)
        new_trk_orig = TrkFile.load(DATA['complex_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)
        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['complex_trk_fname'], 'rb').read()

    def test_load_write_file(self):
        for fname in [DATA['empty_trk_fname'], DATA['simple_trk_fname'], DATA['complex_trk_fname']]:
            for lazy_load in [False, True]:
                trk = TrkFile.load(fname, lazy_load=lazy_load)
                trk_file = BytesIO()
                trk.save(trk_file)
                new_trk = TrkFile.load(fname, lazy_load=False)
                assert_tractogram_equal(new_trk.tractogram, trk.tractogram)

    def test_load_write_LPS_file(self):
        trk_RAS = TrkFile.load(DATA['standard_trk_fname'], lazy_load=False)
        trk_LPS = TrkFile.load(DATA['standard_LPS_trk_fname'], lazy_load=False)
        assert_tractogram_equal(trk_LPS.tractogram, trk_RAS.tractogram)
        trk_file = BytesIO()
        trk = TrkFile(trk_LPS.tractogram, trk_LPS.header)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file)
        assert_arr_dict_equal(new_trk.header, trk.header)
        assert_tractogram_equal(new_trk.tractogram, trk.tractogram)
        new_trk_orig = TrkFile.load(DATA['standard_LPS_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)
        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['standard_LPS_trk_fname'], 'rb').read()
        trk_file = BytesIO()
        header = copy.deepcopy(trk_LPS.header)
        header[Field.VOXEL_ORDER] = b''
        trk = TrkFile(trk_LPS.tractogram, header)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file)
        assert_arr_dict_equal(new_trk.header, trk_LPS.header)
        assert_tractogram_equal(new_trk.tractogram, trk.tractogram)
        new_trk_orig = TrkFile.load(DATA['standard_LPS_trk_fname'])
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)
        trk_file.seek(0, os.SEEK_SET)
        assert trk_file.read() == open(DATA['standard_LPS_trk_fname'], 'rb').read()

    def test_write_optional_header_fields(self):
        tractogram = Tractogram(affine_to_rasmm=np.eye(4))
        trk_file = BytesIO()
        header = {'extra': 1234}
        trk = TrkFile(tractogram, header)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)
        new_trk = TrkFile.load(trk_file)
        assert 'extra' not in new_trk.header

    def test_write_too_many_scalars_and_properties(self):
        data_per_point = {}
        for i in range(10):
            data_per_point[f'#{i}'] = DATA['fa']
            tractogram = Tractogram(DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4))
            trk_file = BytesIO()
            trk = TrkFile(tractogram)
            trk.save(trk_file)
            trk_file.seek(0, os.SEEK_SET)
            new_trk = TrkFile.load(trk_file, lazy_load=False)
            assert_tractogram_equal(new_trk.tractogram, tractogram)
        data_per_point[f'#{i + 1}'] = DATA['fa']
        tractogram = Tractogram(DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4))
        trk = TrkFile(tractogram)
        with pytest.raises(ValueError):
            trk.save(BytesIO())
        data_per_streamline = {}
        for i in range(10):
            data_per_streamline[f'#{i}'] = DATA['mean_torsion']
            tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
            trk_file = BytesIO()
            trk = TrkFile(tractogram)
            trk.save(trk_file)
            trk_file.seek(0, os.SEEK_SET)
            new_trk = TrkFile.load(trk_file, lazy_load=False)
            assert_tractogram_equal(new_trk.tractogram, tractogram)
        data_per_streamline[f'#{i + 1}'] = DATA['mean_torsion']
        tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline)
        trk = TrkFile(tractogram)
        with pytest.raises(ValueError):
            trk.save(BytesIO())

    def test_write_scalars_and_properties_name_too_long(self):
        for nb_chars in range(22):
            data_per_point = {'A' * nb_chars: DATA['colors']}
            tractogram = Tractogram(DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4))
            trk = TrkFile(tractogram)
            if nb_chars > 18:
                with pytest.raises(ValueError):
                    trk.save(BytesIO())
            else:
                trk.save(BytesIO())
            data_per_point = {'A' * nb_chars: DATA['fa']}
            tractogram = Tractogram(DATA['streamlines'], data_per_point=data_per_point, affine_to_rasmm=np.eye(4))
            trk = TrkFile(tractogram)
            if nb_chars > 20:
                with pytest.raises(ValueError):
                    trk.save(BytesIO())
            else:
                trk.save(BytesIO())
        for nb_chars in range(22):
            data_per_streamline = {'A' * nb_chars: DATA['mean_colors']}
            tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
            trk = TrkFile(tractogram)
            if nb_chars > 18:
                with pytest.raises(ValueError):
                    trk.save(BytesIO())
            else:
                trk.save(BytesIO())
            data_per_streamline = {'A' * nb_chars: DATA['mean_torsion']}
            tractogram = Tractogram(DATA['streamlines'], data_per_streamline=data_per_streamline, affine_to_rasmm=np.eye(4))
            trk = TrkFile(tractogram)
            if nb_chars > 20:
                with pytest.raises(ValueError):
                    trk.save(BytesIO())
            else:
                trk.save(BytesIO())

    def test_str(self):
        trk = TrkFile.load(DATA['complex_trk_fname'])
        str(trk)

    def test_header_read_restore(self):
        trk_fname = DATA['simple_trk_fname']
        bio = BytesIO()
        bio.write(b'Along my very merry way')
        hdr_pos = bio.tell()
        hdr_from_fname = TrkFile._read_header(trk_fname)
        with open(trk_fname, 'rb') as fobj:
            bio.write(fobj.read())
        bio.seek(hdr_pos)
        hdr_from_fname['_offset_data'] += hdr_pos
        assert_arr_dict_equal(TrkFile._read_header(bio), hdr_from_fname)
        assert bio.tell() == hdr_pos