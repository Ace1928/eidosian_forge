from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
class TestPythonTracing(TestCase):
    """
    Tests of the interaction of ``sys.settrace()``
    with greenlet facilities.

    NOTE: Most of this is probably CPython specific.
    """
    maxDiff = None

    def test_trace_events_trivial(self):
        with PythonTracer() as actions:
            tpt_callback()
        self.assertEqual(actions, [('return', '__enter__'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('call', '__exit__'), ('c_call', '__exit__')])

    def _trace_switch(self, glet):
        with PythonTracer() as actions:
            glet.switch()
        return actions

    def _check_trace_events_func_already_set(self, glet):
        actions = self._trace_switch(glet)
        self.assertEqual(actions, [('return', '__enter__'), ('c_call', '_trace_switch'), ('call', 'run'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('return', 'run'), ('c_return', '_trace_switch'), ('call', '__exit__'), ('c_call', '__exit__')])

    def test_trace_events_into_greenlet_func_already_set(self):

        def run():
            return tpt_callback()
        self._check_trace_events_func_already_set(greenlet.greenlet(run))

    def test_trace_events_into_greenlet_subclass_already_set(self):

        class X(greenlet.greenlet):

            def run(self):
                return tpt_callback()
        self._check_trace_events_func_already_set(X())

    def _check_trace_events_from_greenlet_sets_profiler(self, g, tracer):
        g.switch()
        tpt_callback()
        tracer.__exit__()
        self.assertEqual(tracer.actions, [('return', '__enter__'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('return', 'run'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('call', '__exit__'), ('c_call', '__exit__')])

    def test_trace_events_from_greenlet_func_sets_profiler(self):
        tracer = PythonTracer()

        def run():
            tracer.__enter__()
            return tpt_callback()
        self._check_trace_events_from_greenlet_sets_profiler(greenlet.greenlet(run), tracer)

    def test_trace_events_from_greenlet_subclass_sets_profiler(self):
        tracer = PythonTracer()

        class X(greenlet.greenlet):

            def run(self):
                tracer.__enter__()
                return tpt_callback()
        self._check_trace_events_from_greenlet_sets_profiler(X(), tracer)

    @unittest.skipIf(*DEBUG_BUILD_PY312)
    def test_trace_events_multiple_greenlets_switching(self):
        tracer = PythonTracer()
        g1 = None
        g2 = None

        def g1_run():
            tracer.__enter__()
            tpt_callback()
            g2.switch()
            tpt_callback()
            return 42

        def g2_run():
            tpt_callback()
            tracer.__exit__()
            tpt_callback()
            g1.switch()
        g1 = greenlet.greenlet(g1_run)
        g2 = greenlet.greenlet(g2_run)
        x = g1.switch()
        self.assertEqual(x, 42)
        tpt_callback()
        self.assertEqual(tracer.actions, [('return', '__enter__'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('c_call', 'g1_run'), ('call', 'g2_run'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('call', '__exit__'), ('c_call', '__exit__')])

    @unittest.skipIf(*DEBUG_BUILD_PY312)
    def test_trace_events_multiple_greenlets_switching_siblings(self):
        tracer = PythonTracer()
        g1 = None
        g2 = None

        def g1_run():
            greenlet.getcurrent().parent.switch()
            tracer.__enter__()
            tpt_callback()
            g2.switch()
            tpt_callback()
            return 42

        def g2_run():
            greenlet.getcurrent().parent.switch()
            tpt_callback()
            tracer.__exit__()
            tpt_callback()
            g1.switch()
        g1 = greenlet.greenlet(g1_run)
        g2 = greenlet.greenlet(g2_run)
        g1.switch()
        g2.switch()
        x = g1.switch()
        self.assertEqual(x, 42)
        tpt_callback()
        self.assertEqual(tracer.actions, [('return', '__enter__'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('c_call', 'g1_run'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('call', '__exit__'), ('c_call', '__exit__')])