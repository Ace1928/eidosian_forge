import os
import contextlib
import functools
import gc
import socket
import sys
import textwrap
import types
import warnings
class _DeprecatedMeta(type):

    def __instancecheck__(self, other):
        warnings.warn(_DEPR_MSG.format(old_name, next_version, new_class.__name__), FutureWarning, stacklevel=2)
        return isinstance(other, new_class)