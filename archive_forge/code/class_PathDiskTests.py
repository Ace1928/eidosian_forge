import io
import pathlib
import unittest
import importlib_resources as resources
from . import data01
from . import util
class PathDiskTests(PathTests, unittest.TestCase):
    data = data01

    def test_natural_path(self):
        """
        Guarantee the internal implementation detail that
        file-system-backed resources do not get the tempdir
        treatment.
        """
        target = resources.files(self.data) / 'utf-8.file'
        with resources.as_file(target) as path:
            assert 'data' in str(path)