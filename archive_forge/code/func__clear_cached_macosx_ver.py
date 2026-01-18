import importlib.util
import os
import re
import string
import subprocess
import sys
import sysconfig
import functools
from .errors import DistutilsPlatformError, DistutilsByteCompileError
from ._modified import newer
from .spawn import spawn
from ._log import log
from distutils.util import byte_compile
def _clear_cached_macosx_ver():
    """For testing only. Do not call."""
    global _syscfg_macosx_ver
    _syscfg_macosx_ver = None