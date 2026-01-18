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
class convoluted(RawGreenlet):

    def __getattribute__(self, name):
        if name == 'run':
            self.parent = another[0]
        return RawGreenlet.__getattribute__(self, name)