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
class TestPathlibSupport(TestCase):
    """
        Check that h5py doesn't break on pathlib
    """

    def test_pathlib_accepted_file(self):
        """ Check that pathlib is accepted by h5py.File """
        with closed_tempfile() as f:
            path = pathlib.Path(f)
            with File(path, 'w') as f2:
                self.assertTrue(True)

    def test_pathlib_name_match(self):
        """ Check that using pathlib does not affect naming """
        with closed_tempfile() as f:
            path = pathlib.Path(f)
            with File(path, 'w') as h5f1:
                pathlib_name = h5f1.filename
            with File(f, 'w') as h5f2:
                normal_name = h5f2.filename
            self.assertEqual(pathlib_name, normal_name)