import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
class ResourceDiskTests(ResourceTests, unittest.TestCase):

    def setUp(self):
        self.data = data01