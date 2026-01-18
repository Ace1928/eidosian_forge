import unittest
from collections import namedtuple
from bpython.curtsies import combined_events
from bpython.test import FixLanguageTestCase as TestCase
import curtsies.events
class TestCurtsiesPasteDetection(TestCase):

    def test_paste_threshold(self):
        eg = EventGenerator(list('abc'))
        cb = combined_events(eg, paste_threshold=3)
        e = next(cb)
        self.assertIsInstance(e, curtsies.events.PasteEvent)
        self.assertEqual(e.events, list('abc'))
        self.assertEqual(next(cb), None)
        eg = EventGenerator(list('abc'))
        cb = combined_events(eg, paste_threshold=4)
        self.assertEqual(next(cb), 'a')
        self.assertEqual(next(cb), 'b')
        self.assertEqual(next(cb), 'c')
        self.assertEqual(next(cb), None)

    def test_set_timeout(self):
        eg = EventGenerator('a', zip('bcdefg', [1, 2, 3, 3, 3, 4]))
        eg.schedule_event(curtsies.events.SigIntEvent(), 5)
        eg.schedule_event('h', 6)
        cb = combined_events(eg, paste_threshold=3)
        self.assertEqual(next(cb), 'a')
        self.assertEqual(cb.send(0), None)
        self.assertEqual(next(cb), 'b')
        self.assertEqual(cb.send(0), None)
        eg.tick()
        self.assertEqual(cb.send(0), 'c')
        self.assertEqual(cb.send(0), None)
        eg.tick()
        self.assertIsInstance(cb.send(0), curtsies.events.PasteEvent)
        self.assertEqual(cb.send(0), None)
        self.assertEqual(cb.send(None), 'g')
        self.assertEqual(cb.send(0), None)
        eg.tick(1)
        self.assertIsInstance(cb.send(0), curtsies.events.SigIntEvent)
        self.assertEqual(cb.send(0), None)
        self.assertEqual(cb.send(None), 'h')
        self.assertEqual(cb.send(None), None)