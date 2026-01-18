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
def _handle_header(self, st, mo, fpname):
    st.sectname = mo.group('header')
    if st.sectname in self._sections:
        if self._strict and st.sectname in st.elements_added:
            raise DuplicateSectionError(st.sectname, fpname, st.lineno)
        st.cursect = self._sections[st.sectname]
        st.elements_added.add(st.sectname)
    elif st.sectname == self.default_section:
        st.cursect = self._defaults
    else:
        st.cursect = self._dict()
        self._sections[st.sectname] = st.cursect
        self._proxies[st.sectname] = SectionProxy(self, st.sectname)
        st.elements_added.add(st.sectname)
    st.optname = None