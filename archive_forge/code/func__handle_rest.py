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
def _handle_rest(self, st, line, fpname):
    if self._allow_unnamed_section and st.cursect is None:
        st.sectname = UNNAMED_SECTION
        st.cursect = self._dict()
        self._sections[st.sectname] = st.cursect
        self._proxies[st.sectname] = SectionProxy(self, st.sectname)
        st.elements_added.add(st.sectname)
    st.indent_level = st.cur_indent_level
    mo = self.SECTCRE.match(line.clean)
    if not mo and st.cursect is None:
        raise MissingSectionHeaderError(fpname, st.lineno, line)
    self._handle_header(st, mo, fpname) if mo else self._handle_option(st, line, fpname)