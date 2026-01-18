import os
import platform
import shutil
import tempfile
import unittest
from gzip import GzipFile
from pathlib import Path
import pytest
from monty.shutil import (
class TestCompressFileDir:

    def setup_method(self):
        with open(os.path.join(test_dir, 'tempfile'), 'w') as f:
            f.write('hello world')

    def test_compress_and_decompress_file(self):
        fname = os.path.join(test_dir, 'tempfile')
        for fmt in ['gz', 'bz2']:
            compress_file(fname, fmt)
            assert os.path.exists(fname + '.' + fmt)
            assert not os.path.exists(fname)
            decompress_file(fname + '.' + fmt)
            assert os.path.exists(fname)
            assert not os.path.exists(fname + '.' + fmt)
        with open(fname) as f:
            txt = f.read()
            assert txt == 'hello world'
        with pytest.raises(ValueError):
            compress_file('whatever', 'badformat')
        assert decompress_file('non-existent') is None
        assert decompress_file('non-existent.gz') is None
        assert decompress_file('non-existent.bz2') is None

    def teardown_method(self):
        os.remove(os.path.join(test_dir, 'tempfile'))