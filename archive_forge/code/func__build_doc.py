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
def _build_doc(self):
    """
        Raises
        ------
        ValueError
            * If a URL that lxml cannot parse is passed.

        Exception
            * Any other ``Exception`` thrown. For example, trying to parse a
              URL that is syntactically correct on a machine with no internet
              connection will fail.

        See Also
        --------
        pandas.io.html._HtmlFrameParser._build_doc
        """
    from lxml.etree import XMLSyntaxError
    from lxml.html import HTMLParser, fromstring, parse
    parser = HTMLParser(recover=True, encoding=self.encoding)
    try:
        if is_url(self.io):
            with get_handle(self.io, 'r', storage_options=self.storage_options) as f:
                r = parse(f.handle, parser=parser)
        else:
            r = parse(self.io, parser=parser)
        try:
            r = r.getroot()
        except AttributeError:
            pass
    except (UnicodeDecodeError, OSError) as e:
        if not is_url(self.io):
            r = fromstring(self.io, parser=parser)
            try:
                r = r.getroot()
            except AttributeError:
                pass
        else:
            raise e
    else:
        if not hasattr(r, 'text_content'):
            raise XMLSyntaxError('no text parsed from document', 0, 0, 0)
    for br in r.xpath('*//br'):
        br.tail = '\n' + (br.tail or '')
    return r