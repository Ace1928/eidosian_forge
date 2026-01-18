import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
@classmethod
def _extract_from_extended_frame_gen(klass, frame_gen, *, limit=None, lookup_lines=True, capture_locals=False):
    if limit is None:
        limit = getattr(sys, 'tracebacklimit', None)
        if limit is not None and limit < 0:
            limit = 0
    if limit is not None:
        if limit >= 0:
            frame_gen = itertools.islice(frame_gen, limit)
        else:
            frame_gen = collections.deque(frame_gen, maxlen=-limit)
    result = klass()
    fnames = set()
    for f, (lineno, end_lineno, colno, end_colno) in frame_gen:
        co = f.f_code
        filename = co.co_filename
        name = co.co_name
        fnames.add(filename)
        linecache.lazycache(filename, f.f_globals)
        if capture_locals:
            f_locals = f.f_locals
        else:
            f_locals = None
        result.append(FrameSummary(filename, lineno, name, lookup_line=False, locals=f_locals, end_lineno=end_lineno, colno=colno, end_colno=end_colno))
    for filename in fnames:
        linecache.checkcache(filename)
    if lookup_lines:
        for f in result:
            f.line
    return result