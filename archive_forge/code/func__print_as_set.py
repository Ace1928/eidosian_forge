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
def _print_as_set(s) -> str:
    arg = ', '.join([pprint_thing(el) for el in s])
    return f'{{{arg}}}'