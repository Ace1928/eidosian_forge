import networkx as nx
import pickle
from taskflow import test
from taskflow.types import graph
from taskflow.types import sets
from taskflow.types import timing
from taskflow.types import tree
class TimingTest(test.TestCase):

    def test_convert_fail(self):
        for baddie in ['abc123', '-1', '', object()]:
            self.assertRaises(ValueError, timing.convert_to_timeout, baddie)

    def test_convert_noop(self):
        t = timing.convert_to_timeout(1.0)
        t2 = timing.convert_to_timeout(t)
        self.assertEqual(t, t2)

    def test_interrupt(self):
        t = timing.convert_to_timeout(1.0)
        self.assertFalse(t.is_stopped())
        t.interrupt()
        self.assertTrue(t.is_stopped())

    def test_reset(self):
        t = timing.convert_to_timeout(1.0)
        t.interrupt()
        self.assertTrue(t.is_stopped())
        t.reset()
        self.assertFalse(t.is_stopped())

    def test_values(self):
        for v, e_v in [('1.0', 1.0), (1, 1.0), ('2.0', 2.0)]:
            t = timing.convert_to_timeout(v)
            self.assertEqual(e_v, t.value)

    def test_fail(self):
        self.assertRaises(ValueError, timing.Timeout, -1)