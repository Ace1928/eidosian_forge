import unittest
import sys
import os
from io import BytesIO
from distutils import cygwinccompiler
from distutils.cygwinccompiler import (check_config_h,
from distutils.tests import support
def _find_executable(self, name):
    if name in self._exes:
        return name
    return None