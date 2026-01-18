from __future__ import annotations
import codecs
import io
from typing import (
import warnings
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import (
def _convert_empty_str_key(self) -> None:
    """
        Replace zero-length string in `namespaces`.

        This method will replace '' with None to align to `lxml`
        requirement that empty string prefixes are not allowed.
        """
    if self.namespaces and '' in self.namespaces.keys():
        self.namespaces[None] = self.namespaces.pop('', 'default')