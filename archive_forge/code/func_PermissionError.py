import collections
import contextlib
import errno
import functools
import os
import sys
import types
@_instance_checking_exception(EnvironmentError)
def PermissionError(inst):
    return getattr(inst, 'errno', _SENTINEL) in (errno.EACCES, errno.EPERM)