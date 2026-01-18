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
class MockingTestCase(unittest.TestCase):
    """
    A TestCase class designed to dynamically add various mockers that should be used in every test, mimicking the
    behavior of a class-wide mock when defining one normally will not do.

    Useful when a mock requires specific information available only initialized after `TestCase.setUpClass`, such as
    setting an environment variable with that information.

    The `add_mocks` function should be ran at the end of a `TestCase`'s `setUp` function, after a call to
    `super().setUp()` such as:
    ```python
    def setUp(self):
        super().setUp()
        mocks = mock.patch.dict(os.environ, {"SOME_ENV_VAR", "SOME_VALUE"})
        self.add_mocks(mocks)
    ```
    """

    def add_mocks(self, mocks: Union[mock.Mock, List[mock.Mock]]):
        """
        Add custom mocks for tests that should be repeated on each test. Should be called during
        `MockingTestCase.setUp`, after `super().setUp()`.

        Args:
            mocks (`mock.Mock` or list of `mock.Mock`):
                Mocks that should be added to the `TestCase` after `TestCase.setUpClass` has been run
        """
        self.mocks = mocks if isinstance(mocks, (tuple, list)) else [mocks]
        for m in self.mocks:
            m.start()
            self.addCleanup(m.stop)