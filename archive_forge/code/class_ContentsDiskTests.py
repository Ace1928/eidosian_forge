import unittest
import importlib_resources as resources
from . import data01
from . import util
class ContentsDiskTests(ContentsTests, unittest.TestCase):

    def setUp(self):
        self.data = data01