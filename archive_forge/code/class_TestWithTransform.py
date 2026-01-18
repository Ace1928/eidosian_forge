import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options
class TestWithTransform(object):

    def test_simplified(self):
        t = self.run_pipeline([WithTransform(None)], u'\n        with x:\n            y = z ** 3\n        ')
        self.assertCode(u'\n\n        $0_0 = x\n        $0_2 = $0_0.__exit__\n        $0_0.__enter__()\n        $0_1 = True\n        try:\n            try:\n                $1_0 = None\n                y = z ** 3\n            except:\n                $0_1 = False\n                if (not $0_2($1_0)):\n                    raise\n        finally:\n            if $0_1:\n                $0_2(None, None, None)\n\n        ', t)

    def test_basic(self):
        t = self.run_pipeline([WithTransform(None)], u'\n        with x as y:\n            y = z ** 3\n        ')
        self.assertCode(u'\n\n        $0_0 = x\n        $0_2 = $0_0.__exit__\n        $0_3 = $0_0.__enter__()\n        $0_1 = True\n        try:\n            try:\n                $1_0 = None\n                y = $0_3\n                y = z ** 3\n            except:\n                $0_1 = False\n                if (not $0_2($1_0)):\n                    raise\n        finally:\n            if $0_1:\n                $0_2(None, None, None)\n\n        ', t)