import unittest
import importlib_resources as resources
from . import data01
from . import util
class ContentsTests:
    expected = {'__init__.py', 'binary.file', 'subdirectory', 'utf-16.file', 'utf-8.file'}

    def test_contents(self):
        contents = {path.name for path in resources.files(self.data).iterdir()}
        assert self.expected <= contents