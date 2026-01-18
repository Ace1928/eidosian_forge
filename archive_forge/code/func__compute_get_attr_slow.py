from __future__ import nested_scopes
import traceback
import warnings
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import _pydev_saved_modules
import signal
import os
import ctypes
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from urllib.parse import quote  # @UnresolvedImport
import time
import inspect
import sys
from _pydevd_bundle.pydevd_constants import USE_CUSTOM_SYS_CURRENT_FRAMES, IS_PYPY, SUPPORT_GEVENT, \
def _compute_get_attr_slow(self, diff, cls, attr_name):
    try:
        cls = cls.__name__
    except:
        pass
    return 'pydevd warning: Getting attribute %s.%s was slow (took %.2fs)\nCustomize report timeout by setting the `PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT` environment variable to a higher timeout (default is: %ss)\n' % (cls, attr_name, diff, PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT)