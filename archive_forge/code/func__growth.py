from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
def _growth(self):
    return objgraph.growth(limit=None, peak_stats=self.peak_stats, filter=self._include_object_p)