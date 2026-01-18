from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import sys
import time
import threading
from abc import ABCMeta, abstractmethod
import greenlet
from greenlet import greenlet as RawGreenlet
from . import TestCase
from .leakcheck import fails_leakcheck
class TestBrokenGreenlets(TestCase):

    def test_failed_to_initialstub(self):

        def func():
            raise AssertionError('Never get here')
        g = greenlet._greenlet.UnswitchableGreenlet(func)
        g.force_switch_error = True
        with self.assertRaisesRegex(SystemError, 'Failed to switch stacks into a greenlet for the first time.'):
            g.switch()

    def test_failed_to_switch_into_running(self):
        runs = []

        def func():
            runs.append(1)
            greenlet.getcurrent().parent.switch()
            runs.append(2)
            greenlet.getcurrent().parent.switch()
            runs.append(3)
        g = greenlet._greenlet.UnswitchableGreenlet(func)
        g.switch()
        self.assertEqual(runs, [1])
        g.switch()
        self.assertEqual(runs, [1, 2])
        g.force_switch_error = True
        with self.assertRaisesRegex(SystemError, 'Failed to switch stacks into a running greenlet.'):
            g.switch()
        g.force_switch_error = False
        g.switch()
        self.assertEqual(runs, [1, 2, 3])

    def test_failed_to_slp_switch_into_running(self):
        ex = self.assertScriptRaises('fail_slp_switch.py')
        self.assertIn('fail_slp_switch is running', ex.output)
        self.assertIn(ex.returncode, self.get_expected_returncodes_for_aborted_process())

    def test_reentrant_switch_two_greenlets(self):
        output = self.run_script('fail_switch_two_greenlets.py')
        self.assertIn('In g1_run', output)
        self.assertIn('TRACE', output)
        self.assertIn('LEAVE TRACE', output)
        self.assertIn('Falling off end of main', output)
        self.assertIn('Falling off end of g1_run', output)
        self.assertIn('Falling off end of g2', output)

    def test_reentrant_switch_three_greenlets(self):
        ex = self.assertScriptRaises('fail_switch_three_greenlets.py', exitcodes=(1,))
        self.assertIn('TypeError', ex.output)
        self.assertIn('positional arguments', ex.output)

    def test_reentrant_switch_three_greenlets2(self):
        output = self.run_script('fail_switch_three_greenlets2.py')
        self.assertIn("RESULTS: [('trace', 'switch'), ('trace', 'switch'), ('g2 arg', 'g2 from tracefunc'), ('trace', 'switch'), ('main g1', 'from g2_run'), ('trace', 'switch'), ('g1 arg', 'g1 from main'), ('trace', 'switch'), ('main g2', 'from g1_run'), ('trace', 'switch'), ('g1 from parent', 'g1 from main 2'), ('trace', 'switch'), ('main g1.2', 'g1 done'), ('trace', 'switch'), ('g2 from parent', ()), ('trace', 'switch'), ('main g2.2', 'g2 done')]", output)

    def test_reentrant_switch_GreenletAlreadyStartedInPython(self):
        output = self.run_script('fail_initialstub_already_started.py')
        self.assertIn("RESULTS: ['Begin C', 'Switch to b from B.__getattribute__ in C', ('Begin B', ()), '_B_run switching to main', ('main from c', 'From B'), 'B.__getattribute__ back from main in C', ('Begin A', (None,)), ('A dead?', True, 'B dead?', True, 'C dead?', False), 'C done', ('main from c.2', None)]", output)

    def test_reentrant_switch_run_callable_has_del(self):
        output = self.run_script('fail_clearing_run_switches.py')
        self.assertIn("RESULTS [('G.__getattribute__', 'run'), ('RunCallable', '__del__'), ('main: g.switch()', 'from RunCallable'), ('run_func', 'enter')]", output)