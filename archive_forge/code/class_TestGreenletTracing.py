from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
class TestGreenletTracing(TestCase):
    """
    Tests of ``greenlet.settrace()``
    """

    def test_a_greenlet_tracing(self):
        main = greenlet.getcurrent()

        def dummy():
            pass

        def dummyexc():
            raise SomeError()
        with GreenletTracer() as actions:
            g1 = greenlet.greenlet(dummy)
            g1.switch()
            g2 = greenlet.greenlet(dummyexc)
            self.assertRaises(SomeError, g2.switch)
        self.assertEqual(actions, [('switch', (main, g1)), ('switch', (g1, main)), ('switch', (main, g2)), ('throw', (g2, main))])

    def test_b_exception_disables_tracing(self):
        main = greenlet.getcurrent()

        def dummy():
            main.switch()
        g = greenlet.greenlet(dummy)
        g.switch()
        with GreenletTracer(error_on_trace=True) as actions:
            self.assertRaises(SomeError, g.switch)
            self.assertEqual(greenlet.gettrace(), None)
        self.assertEqual(actions, [('switch', (main, g))])

    def test_set_same_tracer_twice(self):
        tracer = GreenletTracer()
        with tracer:
            greenlet.settrace(tracer)