import os
import platform
import shutil
import tempfile
import unittest
from gzip import GzipFile
from pathlib import Path
import pytest
from monty.shutil import (
class TestGzipDir:

    def setup_method(self):
        os.mkdir(os.path.join(test_dir, 'gzip_dir'))
        with open(os.path.join(test_dir, 'gzip_dir', 'tempfile'), 'w') as f:
            f.write('what')
        self.mtime = os.path.getmtime(os.path.join(test_dir, 'gzip_dir', 'tempfile'))

    def test_gzip(self):
        full_f = os.path.join(test_dir, 'gzip_dir', 'tempfile')
        gzip_dir(os.path.join(test_dir, 'gzip_dir'))
        assert os.path.exists(f'{full_f}.gz')
        assert not os.path.exists(full_f)
        with GzipFile(f'{full_f}.gz') as g:
            assert g.readline().decode('utf-8') == 'what'
        assert os.path.getmtime(f'{full_f}.gz') == pytest.approx(self.mtime, 4)

    def test_handle_sub_dirs(self):
        sub_dir = os.path.join(test_dir, 'gzip_dir', 'sub_dir')
        sub_file = os.path.join(sub_dir, 'new_tempfile')
        os.mkdir(sub_dir)
        with open(sub_file, 'w') as f:
            f.write('anotherwhat')
        gzip_dir(os.path.join(test_dir, 'gzip_dir'))
        assert os.path.exists(f'{sub_file}.gz')
        assert not os.path.exists(sub_file)
        with GzipFile(f'{sub_file}.gz') as g:
            assert g.readline().decode('utf-8') == 'anotherwhat'

    def teardown_method(self):
        shutil.rmtree(os.path.join(test_dir, 'gzip_dir'))