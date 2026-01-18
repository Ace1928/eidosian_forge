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
def _do_simple_test(self):
    lst = []

    def f():
        lst.append(1)
        greenlet.getcurrent().parent.switch()
        lst.append(3)
    g = RawGreenlet(f)
    lst.append(0)
    g.switch()
    lst.append(2)
    g.switch()
    lst.append(4)
    self.assertEqual(lst, list(range(5)))