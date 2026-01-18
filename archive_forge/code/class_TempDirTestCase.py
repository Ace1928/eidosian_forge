import asyncio
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Union
from unittest import mock
import torch
import accelerate
from ..state import AcceleratorState, PartialState
from ..utils import (
class TempDirTestCase(unittest.TestCase):
    """
    A TestCase class that keeps a single `tempfile.TemporaryDirectory` open for the duration of the class, wipes its
    data at the start of a test, and then destroyes it at the end of the TestCase.

    Useful for when a class or API requires a single constant folder throughout it's use, such as Weights and Biases

    The temporary directory location will be stored in `self.tmpdir`
    """
    clear_on_setup = True

    @classmethod
    def setUpClass(cls):
        """Creates a `tempfile.TemporaryDirectory` and stores it in `cls.tmpdir`"""
        cls.tmpdir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        """Remove `cls.tmpdir` after test suite has finished"""
        if os.path.exists(cls.tmpdir):
            shutil.rmtree(cls.tmpdir)

    def setUp(self):
        """Destroy all contents in `self.tmpdir`, but not `self.tmpdir`"""
        if self.clear_on_setup:
            for path in self.tmpdir.glob('**/*'):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)