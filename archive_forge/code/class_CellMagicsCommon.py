import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
class CellMagicsCommon(object):

    def test_whole_cell(self):
        src = '%%cellm line\nbody\n'
        out = self.sp.transform_cell(src)
        ref = "get_ipython().run_cell_magic('cellm', 'line', 'body')\n"
        assert out == ref

    def test_cellmagic_help(self):
        self.sp.push('%%cellm?')
        assert self.sp.push_accepts_more() is False

    def tearDown(self):
        self.sp.reset()