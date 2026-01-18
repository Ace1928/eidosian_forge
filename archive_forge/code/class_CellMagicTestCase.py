import gc
import io
import os
import re
import shlex
import sys
import warnings
from importlib import invalidate_caches
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest import TestCase, mock
import pytest
from IPython import get_ipython
from IPython.core import magic
from IPython.core.error import UsageError
from IPython.core.magic import (
from IPython.core.magics import code, execution, logging, osm, script
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
from IPython.utils.process import find_cmd
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.syspathcontext import prepended_to_syspath
from .test_debugger import PdbTestInput
from tempfile import NamedTemporaryFile
from IPython.core.magic import (
class CellMagicTestCase(TestCase):

    def check_ident(self, magic):
        out = _ip.run_cell_magic(magic, 'a', 'b')
        assert out == ('a', 'b')
        _ip.run_cell('%%' + magic + ' c\nd\n')
        assert _ip.user_ns['_'] == ('c', 'd\n')

    def test_cell_magic_func_deco(self):
        """Cell magic using simple decorator"""

        @register_cell_magic
        def cellm(line, cell):
            return (line, cell)
        self.check_ident('cellm')

    def test_cell_magic_reg(self):
        """Cell magic manually registered"""

        def cellm(line, cell):
            return (line, cell)
        _ip.register_magic_function(cellm, 'cell', 'cellm2')
        self.check_ident('cellm2')

    def test_cell_magic_class(self):
        """Cell magics declared via a class"""

        @magics_class
        class MyMagics(Magics):

            @cell_magic
            def cellm3(self, line, cell):
                return (line, cell)
        _ip.register_magics(MyMagics)
        self.check_ident('cellm3')

    def test_cell_magic_class2(self):
        """Cell magics declared via a class, #2"""

        @magics_class
        class MyMagics2(Magics):

            @cell_magic('cellm4')
            def cellm33(self, line, cell):
                return (line, cell)
        _ip.register_magics(MyMagics2)
        self.check_ident('cellm4')
        c33 = _ip.find_cell_magic('cellm33')
        assert c33 == None