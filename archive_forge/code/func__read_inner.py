from __future__ import annotations
from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import contextlib
from dataclasses import dataclass, field
import functools
from .compat import io
import itertools
import os
import re
import sys
from typing import Iterable
def _read_inner(self, fp, fpname):
    st = _ReadState()
    Line = functools.partial(_Line, prefixes=self._prefixes)
    for st.lineno, line in enumerate(map(Line, fp), start=1):
        if not line.clean:
            if self._empty_lines_in_values:
                if not line.has_comments and st.cursect is not None and st.optname and (st.cursect[st.optname] is not None):
                    st.cursect[st.optname].append('')
            else:
                st.indent_level = sys.maxsize
            continue
        first_nonspace = self.NONSPACECRE.search(line)
        st.cur_indent_level = first_nonspace.start() if first_nonspace else 0
        if self._handle_continuation_line(st, line, fpname):
            continue
        self._handle_rest(st, line, fpname)
    return st.errors