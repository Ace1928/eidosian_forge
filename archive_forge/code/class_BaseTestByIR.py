import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
class BaseTestByIR(TestCase):
    refprune_bitmask = 0
    prologue = '\ndeclare void @NRT_incref(i8* %ptr)\ndeclare void @NRT_decref(i8* %ptr)\n'

    def check(self, irmod, subgraph_limit=None):
        mod = llvm.parse_assembly(f'{self.prologue}\n{irmod}')
        pm = llvm.ModulePassManager()
        if subgraph_limit is None:
            pm.add_refprune_pass(self.refprune_bitmask)
        else:
            pm.add_refprune_pass(self.refprune_bitmask, subgraph_limit=subgraph_limit)
        before = llvm.dump_refprune_stats()
        pm.run(mod)
        after = llvm.dump_refprune_stats()
        return (mod, after - before)