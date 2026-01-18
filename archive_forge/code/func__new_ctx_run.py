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
def _new_ctx_run(self, *args, **kwargs):
    return copy_context().run(*args, **kwargs)