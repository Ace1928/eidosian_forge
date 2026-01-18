import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
class TestCountdownIter(test.TestCase):

    def test_expected_count(self):
        upper = 100
        it = misc.countdown_iter(upper)
        items = []
        for i in it:
            self.assertEqual(upper, i)
            upper -= 1
            items.append(i)
        self.assertEqual(0, upper)
        self.assertEqual(100, len(items))

    def test_no_count(self):
        it = misc.countdown_iter(0)
        self.assertEqual(0, len(list(it)))
        it = misc.countdown_iter(-1)
        self.assertEqual(0, len(list(it)))

    def test_expected_count_custom_decr(self):
        upper = 100
        it = misc.countdown_iter(upper, decr=2)
        items = []
        for i in it:
            self.assertEqual(upper, i)
            upper -= 2
            items.append(i)
        self.assertEqual(0, upper)
        self.assertEqual(50, len(items))

    def test_invalid_decr(self):
        it = misc.countdown_iter(10, -1)
        self.assertRaises(ValueError, next, it)