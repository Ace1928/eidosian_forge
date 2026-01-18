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
def _handle_continuation_line(self, st, line, fpname):
    is_continue = st.cursect is not None and st.optname and (st.cur_indent_level > st.indent_level)
    if is_continue:
        if st.cursect[st.optname] is None:
            raise MultilineContinuationError(fpname, st.lineno, line)
        st.cursect[st.optname].append(line.clean)
    return is_continue