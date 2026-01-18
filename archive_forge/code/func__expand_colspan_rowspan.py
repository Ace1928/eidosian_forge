from __future__ import annotations
from collections import abc
import numbers
import re
from re import Pattern
from typing import (
import warnings
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import is_list_like
from pandas import isna
from pandas.core.indexes.base import Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.formats.printing import pprint_thing
from pandas.io.parsers import TextParser
def _expand_colspan_rowspan(self, rows, section: Literal['header', 'footer', 'body']):
    """
        Given a list of <tr>s, return a list of text rows.

        Parameters
        ----------
        rows : list of node-like
            List of <tr>s
        section : the section that the rows belong to (header, body or footer).

        Returns
        -------
        list of list
            Each returned row is a list of str text, or tuple (text, link)
            if extract_links is not None.

        Notes
        -----
        Any cell with ``rowspan`` or ``colspan`` will have its contents copied
        to subsequent cells.
        """
    all_texts = []
    text: str | tuple
    remainder: list[tuple[int, str | tuple, int]] = []
    for tr in rows:
        texts = []
        next_remainder = []
        index = 0
        tds = self._parse_td(tr)
        for td in tds:
            while remainder and remainder[0][0] <= index:
                prev_i, prev_text, prev_rowspan = remainder.pop(0)
                texts.append(prev_text)
                if prev_rowspan > 1:
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
                index += 1
            text = _remove_whitespace(self._text_getter(td))
            if self.extract_links in ('all', section):
                href = self._href_getter(td)
                text = (text, href)
            rowspan = int(self._attr_getter(td, 'rowspan') or 1)
            colspan = int(self._attr_getter(td, 'colspan') or 1)
            for _ in range(colspan):
                texts.append(text)
                if rowspan > 1:
                    next_remainder.append((index, text, rowspan - 1))
                index += 1
        for prev_i, prev_text, prev_rowspan in remainder:
            texts.append(prev_text)
            if prev_rowspan > 1:
                next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
        all_texts.append(texts)
        remainder = next_remainder
    while remainder:
        next_remainder = []
        texts = []
        for prev_i, prev_text, prev_rowspan in remainder:
            texts.append(prev_text)
            if prev_rowspan > 1:
                next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
        all_texts.append(texts)
        remainder = next_remainder
    return all_texts