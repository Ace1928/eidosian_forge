import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
def _check_call_order__subtests(self, result, events, expected_events):

    class Foo(Test.LoggingTestCase):

        def test(self):
            super(Foo, self).test()
            for i in [1, 2, 3]:
                with self.subTest(i=i):
                    if i == 1:
                        self.fail('failure')
                    for j in [2, 3]:
                        with self.subTest(j=j):
                            if i * j == 6:
                                raise RuntimeError('raised by Foo.test')
            1 / 0
    Foo(events).run(result)
    self.assertEqual(events, expected_events)