import unittest
import unittest.mock
import os
from pathlib import Path
from bpython.translations import init
class MagicIterMock(unittest.mock.MagicMock):
    __next__ = unittest.mock.Mock(return_value=None)