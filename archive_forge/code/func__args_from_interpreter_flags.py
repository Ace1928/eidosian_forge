import builtins
import errno
import io
import locale
import os
import time
import signal
import sys
import threading
import warnings
import contextlib
from time import monotonic as _time
import types
def _args_from_interpreter_flags():
    """Return a list of command-line arguments reproducing the current
    settings in sys.flags, sys.warnoptions and sys._xoptions."""
    flag_opt_map = {'debug': 'd', 'dont_write_bytecode': 'B', 'no_site': 'S', 'verbose': 'v', 'bytes_warning': 'b', 'quiet': 'q'}
    args = _optim_args_from_interpreter_flags()
    for flag, opt in flag_opt_map.items():
        v = getattr(sys.flags, flag)
        if v > 0:
            args.append('-' + opt * v)
    if sys.flags.isolated:
        args.append('-I')
    else:
        if sys.flags.ignore_environment:
            args.append('-E')
        if sys.flags.no_user_site:
            args.append('-s')
        if sys.flags.safe_path:
            args.append('-P')
    warnopts = sys.warnoptions[:]
    xoptions = getattr(sys, '_xoptions', {})
    bytes_warning = sys.flags.bytes_warning
    dev_mode = sys.flags.dev_mode
    if bytes_warning > 1:
        warnopts.remove('error::BytesWarning')
    elif bytes_warning:
        warnopts.remove('default::BytesWarning')
    if dev_mode:
        warnopts.remove('default')
    for opt in warnopts:
        args.append('-W' + opt)
    if dev_mode:
        args.extend(('-X', 'dev'))
    for opt in ('faulthandler', 'tracemalloc', 'importtime', 'frozen_modules', 'showrefcount', 'utf8'):
        if opt in xoptions:
            value = xoptions[opt]
            if value is True:
                arg = opt
            else:
                arg = '%s=%s' % (opt, value)
            args.extend(('-X', arg))
    return args