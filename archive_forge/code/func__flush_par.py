import typing as t
from contextlib import contextmanager
from gettext import gettext as _
from ._compat import term_len
from .parser import split_opt
def _flush_par() -> None:
    if not buf:
        return
    if buf[0].strip() == '\x08':
        p.append((indent or 0, True, '\n'.join(buf[1:])))
    else:
        p.append((indent or 0, False, ' '.join(buf)))
    del buf[:]