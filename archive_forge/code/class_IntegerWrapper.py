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
class IntegerWrapper(ast.NodeTransformer):
    """Wraps all integers in a call to Integer()"""

    def visit_Num(self, node):
        if isinstance(node.n, int):
            return ast.Call(func=ast.Name(id='Integer', ctx=ast.Load()), args=[node], keywords=[])
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return self.visit_Num(node)
        return node