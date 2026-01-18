import sys
import unittest
import os
from distutils.errors import DistutilsPlatformError
from distutils.tests import support
def _find_vcvarsall(plat_spec):
    return (None, None)