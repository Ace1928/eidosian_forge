import textwrap
import unittest
import warnings
import importlib
import contextlib
import importlib_resources as resources
from ..abc import Traversable
from . import data01
from . import util
from . import _path
from .compat.py39 import os_helper
from .compat.py312 import import_helper
class FilesTests:

    def test_read_bytes(self):
        files = resources.files(self.data)
        actual = files.joinpath('utf-8.file').read_bytes()
        assert actual == b'Hello, UTF-8 world!\n'

    def test_read_text(self):
        files = resources.files(self.data)
        actual = files.joinpath('utf-8.file').read_text(encoding='utf-8')
        assert actual == 'Hello, UTF-8 world!\n'

    def test_traversable(self):
        assert isinstance(resources.files(self.data), Traversable)

    def test_joinpath_with_multiple_args(self):
        files = resources.files(self.data)
        binfile = files.joinpath('subdirectory', 'binary.file')
        self.assertTrue(binfile.is_file())

    def test_old_parameter(self):
        """
        Files used to take a 'package' parameter. Make sure anyone
        passing by name is still supported.
        """
        with suppress_known_deprecation():
            resources.files(package=self.data)