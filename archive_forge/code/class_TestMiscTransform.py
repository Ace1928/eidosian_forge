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
class TestMiscTransform(unittest.TestCase):

    def test_transform_only_once(self):
        cleanup = 0
        line_t = 0

        def count_cleanup(lines):
            nonlocal cleanup
            cleanup += 1
            return lines

        def count_line_t(lines):
            nonlocal line_t
            line_t += 1
            return lines
        ip.input_transformer_manager.cleanup_transforms.append(count_cleanup)
        ip.input_transformer_manager.line_transforms.append(count_line_t)
        ip.run_cell('1')
        assert cleanup == 1
        assert line_t == 1