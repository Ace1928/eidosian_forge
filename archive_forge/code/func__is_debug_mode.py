import collections.abc
import inspect
import os
import sys
import traceback
import types
def _is_debug_mode():
    return sys.flags.dev_mode or (not sys.flags.ignore_environment and bool(os.environ.get('PYTHONASYNCIODEBUG')))