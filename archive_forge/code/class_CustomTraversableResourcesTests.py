import unittest
import contextlib
import pathlib
import importlib_resources as resources
from .. import abc
from ..abc import TraversableResources, ResourceReader
from . import util
from .compat.py39 import os_helper
class CustomTraversableResourcesTests(unittest.TestCase):

    def setUp(self):
        self.fixtures = contextlib.ExitStack()
        self.addCleanup(self.fixtures.close)

    def test_custom_loader(self):
        temp_dir = pathlib.Path(self.fixtures.enter_context(os_helper.temp_dir()))
        loader = SimpleLoader(MagicResources(temp_dir))
        pkg = util.create_package_from_loader(loader)
        files = resources.files(pkg)
        assert isinstance(files, abc.Traversable)
        assert list(files.iterdir()) == []