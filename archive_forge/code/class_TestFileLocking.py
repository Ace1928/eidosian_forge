import pytest
import os
import stat
import pickle
import tempfile
import subprocess
import sys
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
from h5py._hl.files import direct_vfd
from h5py import File
import h5py
from .. import h5
import pathlib
import sys
import h5py
@pytest.mark.skipif(h5py.version.hdf5_version_tuple < (1, 12, 1) and (h5py.version.hdf5_version_tuple[:2] != (1, 10) or h5py.version.hdf5_version_tuple[2] < 7), reason='Requires HDF5 >= 1.12.1 or 1.10.x >= 1.10.7')
@pytest.mark.skipif('HDF5_USE_FILE_LOCKING' in os.environ, reason='HDF5_USE_FILE_LOCKING env. var. is set')
class TestFileLocking:
    """Test h5py.File file locking option"""

    def test_reopen(self, tmp_path):
        """Test file locking when opening twice the same file"""
        fname = tmp_path / 'test.h5'
        with h5py.File(fname, mode='w', locking=True) as f:
            f.flush()
            with pytest.raises(OSError):
                with h5py.File(fname, mode='r', locking=False) as h5f_read:
                    pass
            with h5py.File(fname, mode='r', locking=True) as h5f_read:
                pass
            with h5py.File(fname, mode='r', locking='best-effort') as h5f_read:
                pass

    def test_unsupported_locking(self, tmp_path):
        """Test with erroneous file locking value"""
        fname = tmp_path / 'test.h5'
        with pytest.raises(ValueError):
            with h5py.File(fname, mode='r', locking='unsupported-value') as h5f_read:
                pass

    def test_multiprocess(self, tmp_path):
        """Test file locking option from different concurrent processes"""
        fname = tmp_path / 'test.h5'

        def open_in_subprocess(filename, mode, locking):
            """Open HDF5 file in a subprocess and return True on success"""
            h5py_import_dir = str(pathlib.Path(h5py.__file__).parent.parent)
            process = subprocess.run([sys.executable, '-c', f'\nimport sys\nsys.path.insert(0, {h5py_import_dir!r})\nimport h5py\nf = h5py.File({str(filename)!r}, mode={mode!r}, locking={locking})\n                    '], capture_output=True)
            return process.returncode == 0 and (not process.stderr)
        with h5py.File(fname, mode='w', locking=True) as f:
            f['data'] = 1
        with h5py.File(fname, mode='r', locking=False) as f:
            assert open_in_subprocess(fname, mode='w', locking=True)