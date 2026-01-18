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
def _parser_dispatch(flavor: HTMLFlavors | None) -> type[_HtmlFrameParser]:
    """
    Choose the parser based on the input flavor.

    Parameters
    ----------
    flavor : {{"lxml", "html5lib", "bs4"}} or None
        The type of parser to use. This must be a valid backend.

    Returns
    -------
    cls : _HtmlFrameParser subclass
        The parser class based on the requested input flavor.

    Raises
    ------
    ValueError
        * If `flavor` is not a valid backend.
    ImportError
        * If you do not have the requested `flavor`
    """
    valid_parsers = list(_valid_parsers.keys())
    if flavor not in valid_parsers:
        raise ValueError(f'{repr(flavor)} is not a valid flavor, valid flavors are {valid_parsers}')
    if flavor in ('bs4', 'html5lib'):
        import_optional_dependency('html5lib')
        import_optional_dependency('bs4')
    else:
        import_optional_dependency('lxml.etree')
    return _valid_parsers[flavor]