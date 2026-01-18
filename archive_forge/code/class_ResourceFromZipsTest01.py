import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
class ResourceFromZipsTest01(util.ZipSetupBase, unittest.TestCase):
    ZIP_MODULE = 'data01'

    def test_is_submodule_resource(self):
        submodule = import_module('data01.subdirectory')
        self.assertTrue(resources.files(submodule).joinpath('binary.file').is_file())

    def test_read_submodule_resource_by_name(self):
        self.assertTrue(resources.files('data01.subdirectory').joinpath('binary.file').is_file())

    def test_submodule_contents(self):
        submodule = import_module('data01.subdirectory')
        self.assertEqual(names(resources.files(submodule)), {'__init__.py', 'binary.file'})

    def test_submodule_contents_by_name(self):
        self.assertEqual(names(resources.files('data01.subdirectory')), {'__init__.py', 'binary.file'})

    def test_as_file_directory(self):
        with resources.as_file(resources.files('data01')) as data:
            assert data.name == 'data01'
            assert data.is_dir()
            assert data.joinpath('subdirectory').is_dir()
            assert len(list(data.iterdir()))
        assert not data.parent.exists()