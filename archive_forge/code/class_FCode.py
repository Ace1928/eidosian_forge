from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict
class FCode(object):

    def __init__(self, name, filename):
        self.co_name = name
        self.co_filename = filename
        self.co_firstlineno = 1
        self.co_flags = 0