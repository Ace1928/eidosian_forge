import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
class ResourceLoaderTests(unittest.TestCase):

    def test_resource_contents(self):
        package = util.create_package(file=data01, path=data01.__file__, contents=['A', 'B', 'C'])
        self.assertEqual(names(resources.files(package)), {'A', 'B', 'C'})

    def test_is_file(self):
        package = util.create_package(file=data01, path=data01.__file__, contents=['A', 'B', 'C', 'D/E', 'D/F'])
        self.assertTrue(resources.files(package).joinpath('B').is_file())

    def test_is_dir(self):
        package = util.create_package(file=data01, path=data01.__file__, contents=['A', 'B', 'C', 'D/E', 'D/F'])
        self.assertTrue(resources.files(package).joinpath('D').is_dir())

    def test_resource_missing(self):
        package = util.create_package(file=data01, path=data01.__file__, contents=['A', 'B', 'C', 'D/E', 'D/F'])
        self.assertFalse(resources.files(package).joinpath('Z').is_file())