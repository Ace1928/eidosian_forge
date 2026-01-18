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
class TestFlush(TestCase):
    """
        Feature: Files can be flushed
    """

    def test_flush(self):
        """ Flush via .flush method """
        fid = File(self.mktemp(), 'w')
        fid.flush()
        fid.close()