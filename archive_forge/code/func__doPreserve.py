from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def _doPreserve(self, key):
    if not self.preserve and (isinstance(key, bytes) or isinstance(key, str)):
        return key.lower()
    else:
        return key