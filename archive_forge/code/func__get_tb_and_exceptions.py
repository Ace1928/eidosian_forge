import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
def _get_tb_and_exceptions(self, tb_or_exc):
    """
            Given a tracecack or an exception, return a tuple of chained exceptions
            and current traceback to inspect.
            This will deal with selecting the right ``__cause__`` or ``__context__``
            as well as handling cycles, and return a flattened list of exceptions we
            can jump to with do_exceptions.
            """
    _exceptions = []
    if isinstance(tb_or_exc, BaseException):
        traceback, current = (tb_or_exc.__traceback__, tb_or_exc)
        while current is not None:
            if current in _exceptions:
                break
            _exceptions.append(current)
            if current.__cause__ is not None:
                current = current.__cause__
            elif current.__context__ is not None and (not current.__suppress_context__):
                current = current.__context__
            if len(_exceptions) >= self.MAX_CHAINED_EXCEPTION_DEPTH:
                self.message(f'More than {self.MAX_CHAINED_EXCEPTION_DEPTH} chained exceptions found, not all exceptionswill be browsable with `exceptions`.')
                break
    else:
        traceback = tb_or_exc
    return (tuple(reversed(_exceptions)), traceback)