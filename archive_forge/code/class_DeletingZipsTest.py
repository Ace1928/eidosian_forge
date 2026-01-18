import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
class DeletingZipsTest(util.ZipSetupBase, unittest.TestCase):
    """Having accessed resources in a zip file should not keep an open
    reference to the zip.
    """

    def test_iterdir_does_not_keep_open(self):
        [item.name for item in resources.files('data01').iterdir()]

    def test_is_file_does_not_keep_open(self):
        resources.files('data01').joinpath('binary.file').is_file()

    def test_is_file_failure_does_not_keep_open(self):
        resources.files('data01').joinpath('not-present').is_file()

    @unittest.skip('Desired but not supported.')
    def test_as_file_does_not_keep_open(self):
        resources.as_file(resources.files('data01') / 'binary.file')

    def test_entered_path_does_not_keep_open(self):
        """
        Mimic what certifi does on import to make its bundle
        available for the process duration.
        """
        resources.as_file(resources.files('data01') / 'binary.file').__enter__()

    def test_read_binary_does_not_keep_open(self):
        resources.files('data01').joinpath('binary.file').read_bytes()

    def test_read_text_does_not_keep_open(self):
        resources.files('data01').joinpath('utf-8.file').read_text(encoding='utf-8')