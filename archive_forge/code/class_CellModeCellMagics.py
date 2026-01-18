import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
class CellModeCellMagics(CellMagicsCommon, unittest.TestCase):
    sp = isp.IPythonInputSplitter(line_input_checker=False)

    def test_incremental(self):
        sp = self.sp
        sp.push('%%cellm firstline\n')
        assert sp.push_accepts_more() is True
        sp.push('line2\n')
        assert sp.push_accepts_more() is True
        sp.push('\n')
        assert sp.push_accepts_more() is True

    def test_no_strip_coding(self):
        src = '\n'.join(['%%writefile foo.py', '# coding: utf-8', 'print(u"üñîçø∂é")'])
        out = self.sp.transform_cell(src)
        assert '# coding: utf-8' in out