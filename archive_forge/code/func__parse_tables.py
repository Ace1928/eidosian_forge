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
def _parse_tables(self, document, match, kwargs):
    pattern = match.pattern
    xpath_expr = f'//table[.//text()[re:test(., {repr(pattern)})]]'
    if kwargs:
        xpath_expr += _build_xpath_expr(kwargs)
    tables = document.xpath(xpath_expr, namespaces=_re_namespace)
    tables = self._handle_hidden_tables(tables, 'attrib')
    if self.displayed_only:
        for table in tables:
            for elem in table.xpath('.//style'):
                elem.drop_tree()
            for elem in table.xpath('.//*[@style]'):
                if 'display:none' in elem.attrib.get('style', '').replace(' ', ''):
                    elem.drop_tree()
    if not tables:
        raise ValueError(f'No tables found matching regex {repr(pattern)}')
    return tables