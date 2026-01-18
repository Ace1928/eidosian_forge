from __future__ import print_function
import os
import sys
import gc
from functools import wraps
import unittest
import objgraph
def _growth_after(self):
    if 'urlparse' in sys.modules:
        sys.modules['urlparse'].clear_cache()
    if 'urllib.parse' in sys.modules:
        sys.modules['urllib.parse'].clear_cache()
    return self._growth()