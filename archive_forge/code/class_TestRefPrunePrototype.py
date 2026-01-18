import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
class TestRefPrunePrototype(TestCase):
    """
    Test that the prototype is working.
    """

    def generate_test(self, case_gen):
        nodes, edges, expected = case_gen()
        got = proto.FanoutAlgorithm(nodes, edges).run()
        self.assertEqual(expected, got)
    for name, case in _iterate_cases(generate_test):
        locals()[name] = case