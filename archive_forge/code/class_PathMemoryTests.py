import io
import pathlib
import unittest
import importlib_resources as resources
from . import data01
from . import util
class PathMemoryTests(PathTests, unittest.TestCase):

    def setUp(self):
        file = io.BytesIO(b'Hello, UTF-8 world!\n')
        self.addCleanup(file.close)
        self.data = util.create_package(file=file, path=FileNotFoundError('package exists only in memory'))
        self.data.__spec__.origin = None
        self.data.__spec__.has_location = False