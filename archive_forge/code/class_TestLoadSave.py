import os
import tempfile
import unittest
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.compat.py3k import asbytes
import nibabel as nib
from nibabel.testing import clear_and_catch_warnings, data_path, error_warnings
from nibabel.tmpdirs import InTemporaryDirectory
from .. import FORMATS, trk
from ..tractogram import LazyTractogram, Tractogram
from ..tractogram_file import ExtensionWarning, TractogramFile
from .test_tractogram import assert_tractogram_equal
class TestLoadSave(unittest.TestCase):

    def test_load_empty_file(self):
        for lazy_load in [False, True]:
            for empty_filename in DATA['empty_filenames']:
                tfile = nib.streamlines.load(empty_filename, lazy_load=lazy_load)
                assert isinstance(tfile, TractogramFile)
                if lazy_load:
                    assert type(tfile.tractogram), Tractogram
                else:
                    assert type(tfile.tractogram), LazyTractogram
                with pytest.warns(Warning) if lazy_load else error_warnings():
                    assert_tractogram_equal(tfile.tractogram, DATA['empty_tractogram'])

    def test_load_simple_file(self):
        for lazy_load in [False, True]:
            for simple_filename in DATA['simple_filenames']:
                tfile = nib.streamlines.load(simple_filename, lazy_load=lazy_load)
                assert isinstance(tfile, TractogramFile)
                if lazy_load:
                    assert type(tfile.tractogram), Tractogram
                else:
                    assert type(tfile.tractogram), LazyTractogram
                with pytest.warns(Warning) if lazy_load else error_warnings():
                    assert_tractogram_equal(tfile.tractogram, DATA['simple_tractogram'])

    def test_load_complex_file(self):
        for lazy_load in [False, True]:
            for complex_filename in DATA['complex_filenames']:
                tfile = nib.streamlines.load(complex_filename, lazy_load=lazy_load)
                assert isinstance(tfile, TractogramFile)
                if lazy_load:
                    assert type(tfile.tractogram), Tractogram
                else:
                    assert type(tfile.tractogram), LazyTractogram
                tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
                if tfile.SUPPORTS_DATA_PER_POINT:
                    tractogram.data_per_point = DATA['data_per_point']
                if tfile.SUPPORTS_DATA_PER_STREAMLINE:
                    data = DATA['data_per_streamline']
                    tractogram.data_per_streamline = data
                with pytest.warns(Warning) if lazy_load else error_warnings():
                    assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_tractogram_file(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
        trk_file = trk.TrkFile(tractogram)
        with self.assertRaises(ValueError):
            nib.streamlines.save(trk_file, 'dummy.trk', header={})
        with pytest.warns(ExtensionWarning, match='extension'):
            trk_file = trk.TrkFile(tractogram)
            with self.assertRaises(ValueError):
                nib.streamlines.save(trk_file, 'dummy.tck', header={})
        with InTemporaryDirectory():
            nib.streamlines.save(trk_file, 'dummy.trk')
            tfile = nib.streamlines.load('dummy.trk', lazy_load=False)
            assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_empty_file(self):
        tractogram = Tractogram(affine_to_rasmm=np.eye(4))
        for ext, cls in FORMATS.items():
            with InTemporaryDirectory():
                filename = 'streamlines' + ext
                nib.streamlines.save(tractogram, filename)
                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_simple_file(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
        for ext, cls in FORMATS.items():
            with InTemporaryDirectory():
                filename = 'streamlines' + ext
                nib.streamlines.save(tractogram, filename)
                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_complex_file(self):
        complex_tractogram = Tractogram(DATA['streamlines'], DATA['data_per_streamline'], DATA['data_per_point'], affine_to_rasmm=np.eye(4))
        for ext, cls in FORMATS.items():
            with InTemporaryDirectory():
                filename = 'streamlines' + ext
                nb_expected_warnings = (not cls.SUPPORTS_DATA_PER_POINT) + (not cls.SUPPORTS_DATA_PER_STREAMLINE)
                with clear_and_catch_warnings() as w:
                    warnings.simplefilter('always')
                    nib.streamlines.save(complex_tractogram, filename)
                assert len(w) == nb_expected_warnings
                tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
                if cls.SUPPORTS_DATA_PER_POINT:
                    tractogram.data_per_point = DATA['data_per_point']
                if cls.SUPPORTS_DATA_PER_STREAMLINE:
                    data = DATA['data_per_streamline']
                    tractogram.data_per_streamline = data
                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_sliced_tractogram(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
        original_tractogram = tractogram.copy()
        for ext, cls in FORMATS.items():
            with InTemporaryDirectory():
                filename = 'streamlines' + ext
                nib.streamlines.save(tractogram[::2], filename)
                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram[::2])
                assert_tractogram_equal(tractogram, original_tractogram)

    def test_load_unknown_format(self):
        with self.assertRaises(ValueError):
            nib.streamlines.load('')

    def test_save_unknown_format(self):
        with self.assertRaises(ValueError):
            nib.streamlines.save(Tractogram(), '')

    def test_save_from_generator(self):
        tractogram = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
        for ext, _ in FORMATS.items():
            filtered = (s for s in tractogram.streamlines if True)
            lazy_tractogram = LazyTractogram(lambda: filtered, affine_to_rasmm=np.eye(4))
            with InTemporaryDirectory():
                filename = 'streamlines' + ext
                nib.streamlines.save(lazy_tractogram, filename)
                tfile = nib.streamlines.load(filename, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)