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
class StringRejector(ast.NodeTransformer):
    """Throws an InputRejected when it sees a string literal.

    Used to verify that NodeTransformers can signal that a piece of code should
    not be executed by throwing an InputRejected.
    """

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            raise InputRejected('test')
        return node