from __future__ import annotations
import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from markupsafe import escape
from ..utils import cached_property
from .console import Console
def _process_traceback(exc: BaseException, te: traceback.TracebackException | None=None, *, skip: int=0, hide: bool=True) -> traceback.TracebackException:
    if te is None:
        te = traceback.TracebackException.from_exception(exc, lookup_lines=False)
    frame_gen = traceback.walk_tb(exc.__traceback__)
    limit = getattr(sys, 'tracebacklimit', None)
    if limit is not None:
        if limit < 0:
            limit = 0
        frame_gen = itertools.islice(frame_gen, limit)
    if skip:
        frame_gen = itertools.islice(frame_gen, skip, None)
        del te.stack[:skip]
    new_stack: list[DebugFrameSummary] = []
    hidden = False
    for (f, _), fs in zip(frame_gen, te.stack):
        if hide:
            hide_value = f.f_locals.get('__traceback_hide__', False)
            if hide_value in {'before', 'before_and_this'}:
                new_stack = []
                hidden = False
                if hide_value == 'before_and_this':
                    continue
            elif hide_value in {'reset', 'reset_and_this'}:
                hidden = False
                if hide_value == 'reset_and_this':
                    continue
            elif hide_value in {'after', 'after_and_this'}:
                hidden = True
                if hide_value == 'after_and_this':
                    continue
            elif hide_value or hidden:
                continue
        frame_args: dict[str, t.Any] = {'filename': fs.filename, 'lineno': fs.lineno, 'name': fs.name, 'locals': f.f_locals, 'globals': f.f_globals}
        if hasattr(fs, 'colno'):
            frame_args['colno'] = fs.colno
            frame_args['end_colno'] = fs.end_colno
        new_stack.append(DebugFrameSummary(**frame_args))
    while new_stack:
        module = new_stack[0].global_ns.get('__name__')
        if module is None:
            module = new_stack[0].local_ns.get('__name__')
        if module == 'codeop':
            del new_stack[0]
        else:
            break
    te.stack[:] = new_stack
    if te.__context__:
        context_exc = t.cast(BaseException, exc.__context__)
        te.__context__ = _process_traceback(context_exc, te.__context__, hide=hide)
    if te.__cause__:
        cause_exc = t.cast(BaseException, exc.__cause__)
        te.__cause__ = _process_traceback(cause_exc, te.__cause__, hide=hide)
    return te