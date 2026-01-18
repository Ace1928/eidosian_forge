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
class Negator(ast.NodeTransformer):
    """Negates all number literals in an AST."""

    def visit_Num(self, node):
        node.n = -node.n
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return self.visit_Num(node)
        return node