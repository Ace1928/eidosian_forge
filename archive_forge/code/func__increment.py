from __future__ import print_function
import gc
import sys
import unittest
from functools import partial
from unittest import skipUnless
from unittest import skipIf
from greenlet import greenlet
from greenlet import getcurrent
from . import TestCase
def _increment(self, greenlet_id, callback, counts, expect):
    ctx_var = ID_VAR
    if expect is None:
        self.assertIsNone(ctx_var.get())
    else:
        self.assertEqual(ctx_var.get(), expect)
    ctx_var.set(greenlet_id)
    for _ in range(2):
        counts[ctx_var.get()] += 1
        callback()