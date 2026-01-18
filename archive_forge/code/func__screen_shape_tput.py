import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def _screen_shape_tput(*_):
    """cygwin xterm (windows)"""
    try:
        import shlex
        from subprocess import check_call
        return [int(check_call(shlex.split('tput ' + i))) - 1 for i in ('cols', 'lines')]
    except Exception:
        pass
    return (None, None)