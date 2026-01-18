import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
def _fix_raw_path(rstr):
    if config.IS_WIN32:
        rstr = rstr.replace('/', '\\\\\\\\')
    return rstr