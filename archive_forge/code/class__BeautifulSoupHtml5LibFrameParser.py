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
class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    """
    HTML to DataFrame parser that uses BeautifulSoup under the hood.

    See Also
    --------
    pandas.io.html._HtmlFrameParser
    pandas.io.html._LxmlFrameParser

    Notes
    -----
    Documentation strings for this class are in the base class
    :class:`pandas.io.html._HtmlFrameParser`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from bs4 import SoupStrainer
        self._strainer = SoupStrainer('table')

    def _parse_tables(self, document, match, attrs):
        element_name = self._strainer.name
        tables = document.find_all(element_name, attrs=attrs)
        if not tables:
            raise ValueError('No tables found')
        result = []
        unique_tables = set()
        tables = self._handle_hidden_tables(tables, 'attrs')
        for table in tables:
            if self.displayed_only:
                for elem in table.find_all('style'):
                    elem.decompose()
                for elem in table.find_all(style=re.compile('display:\\s*none')):
                    elem.decompose()
            if table not in unique_tables and table.find(string=match) is not None:
                result.append(table)
            unique_tables.add(table)
        if not result:
            raise ValueError(f'No tables found matching pattern {repr(match.pattern)}')
        return result

    def _href_getter(self, obj) -> str | None:
        a = obj.find('a', href=True)
        return None if not a else a['href']

    def _text_getter(self, obj):
        return obj.text

    def _equals_tag(self, obj, tag) -> bool:
        return obj.name == tag

    def _parse_td(self, row):
        return row.find_all(('td', 'th'), recursive=False)

    def _parse_thead_tr(self, table):
        return table.select('thead tr')

    def _parse_tbody_tr(self, table):
        from_tbody = table.select('tbody tr')
        from_root = table.find_all('tr', recursive=False)
        return from_tbody + from_root

    def _parse_tfoot_tr(self, table):
        return table.select('tfoot tr')

    def _setup_build_doc(self):
        raw_text = _read(self.io, self.encoding, self.storage_options)
        if not raw_text:
            raise ValueError(f'No text parsed from document: {self.io}')
        return raw_text

    def _build_doc(self):
        from bs4 import BeautifulSoup
        bdoc = self._setup_build_doc()
        if isinstance(bdoc, bytes) and self.encoding is not None:
            udoc = bdoc.decode(self.encoding)
            from_encoding = None
        else:
            udoc = bdoc
            from_encoding = self.encoding
        soup = BeautifulSoup(udoc, features='html5lib', from_encoding=from_encoding)
        for br in soup.find_all('br'):
            br.replace_with('\n' + br.text)
        return soup