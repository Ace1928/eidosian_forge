import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
class CompatibilityFilesNoReaderTests(unittest.TestCase):

    @property
    def package(self):
        return util.create_package_from_loader(None)

    @property
    def files(self):
        return resources.files(self.package)

    def test_spec_path_joinpath(self):
        self.assertIsInstance(self.files / 'a', CompatibilityFiles.OrphanPath)