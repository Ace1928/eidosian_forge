from __future__ import annotations
import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only
from markupsafe import escape
def _sequence_repr_maker(left: str, right: str, base: t.Type, limit: int=8) -> t.Callable[[DebugReprGenerator, t.Iterable, bool], str]:

    def proxy(self: DebugReprGenerator, obj: t.Iterable, recursive: bool) -> str:
        if recursive:
            return _add_subclass_info(f'{left}...{right}', obj, base)
        buf = [left]
        have_extended_section = False
        for idx, item in enumerate(obj):
            if idx:
                buf.append(', ')
            if idx == limit:
                buf.append('<span class="extended">')
                have_extended_section = True
            buf.append(self.repr(item))
        if have_extended_section:
            buf.append('</span>')
        buf.append(right)
        return _add_subclass_info(''.join(buf), obj, base)
    return proxy