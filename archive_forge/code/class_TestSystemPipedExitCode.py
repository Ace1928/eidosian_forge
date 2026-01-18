import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
class TestSystemPipedExitCode(ExitCodeChecks):

    def setUp(self):
        super().setUp()
        self.system = ip.system_piped

    @skip_win32
    def test_exit_code_ok(self):
        ExitCodeChecks.test_exit_code_ok(self)

    @skip_win32
    def test_exit_code_error(self):
        ExitCodeChecks.test_exit_code_error(self)

    @skip_win32
    def test_exit_code_signal(self):
        ExitCodeChecks.test_exit_code_signal(self)