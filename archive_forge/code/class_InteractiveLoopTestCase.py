import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
class InteractiveLoopTestCase(unittest.TestCase):
    """Tests for an interactive loop like a python shell.
    """

    def check_ns(self, lines, ns):
        """Validate that the given input lines produce the resulting namespace.

        Note: the input lines are given exactly as they would be typed in an
        auto-indenting environment, as mini_interactive_loop above already does
        auto-indenting and prepends spaces to the input.
        """
        src = mini_interactive_loop(pseudo_input(lines))
        test_ns = {}
        exec(src, test_ns)
        for k, v in ns.items():
            self.assertEqual(test_ns[k], v)

    def test_simple(self):
        self.check_ns(['x=1'], dict(x=1))

    def test_simple2(self):
        self.check_ns(['if 1:', 'x=2'], dict(x=2))

    def test_xy(self):
        self.check_ns(['x=1; y=2'], dict(x=1, y=2))

    def test_abc(self):
        self.check_ns(['if 1:', 'a=1', 'b=2', 'c=3'], dict(a=1, b=2, c=3))

    def test_multi(self):
        self.check_ns(['x =(1+', '1+', '2)'], dict(x=4))