import unittest
import warnings
from contextlib import contextmanager
import numpy as np
import llvmlite.binding as llvm
from numba import njit, types
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.codegen import _parse_refprune_flags
from numba.tests.support import override_config, TestCase
class TestRefPruneFlags(TestCase):

    def setUp(self):
        warnings.simplefilter('error', NumbaInvalidConfigWarning)

    def tearDown(self):
        warnings.resetwarnings()

    def test_warn_invalid_flags(self):
        with set_refprune_flags('abc,per_bb,cde'):
            with self.assertWarns(NumbaInvalidConfigWarning) as cm:
                optval = _parse_refprune_flags()
            self.assertEqual(len(cm.warnings), 2)
            self.assertIn('abc', str(cm.warnings[0].message))
            self.assertIn('cde', str(cm.warnings[1].message))
            self.assertEqual(optval, llvm.RefPruneSubpasses.PER_BB)

    def test_valid_flag(self):
        with set_refprune_flags('per_bb, diamond, fanout,fanout_raise'):
            optval = _parse_refprune_flags()
            self.assertEqual(optval, llvm.RefPruneSubpasses.ALL)

    def test_the_all_flag(self):
        with set_refprune_flags('all'):
            optval = _parse_refprune_flags()
            self.assertEqual(optval, llvm.RefPruneSubpasses.ALL)

    def test_some_flags(self):
        with set_refprune_flags('per_bb, fanout'):
            optval = _parse_refprune_flags()
            enumcls = llvm.RefPruneSubpasses
            self.assertEqual(optval, enumcls.PER_BB | enumcls.FANOUT)