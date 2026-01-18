import collections
import contextlib
import errno
import functools
import os
import sys
import types
@_instance_checking_exception(EnvironmentError)
def ProcessLookupError(inst):
    return getattr(inst, 'errno', _SENTINEL) == errno.ESRCH