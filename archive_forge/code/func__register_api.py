import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def _register_api(self):
    for f, (restype, argtypes) in self._API.items():
        func = getattr(self.lib, f)
        func.restype = restype
        func.argtypes = argtypes