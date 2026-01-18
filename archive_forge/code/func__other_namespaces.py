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
@final
def _other_namespaces(self) -> dict:
    """
        Define other namespaces.

        This method will build dictionary of namespaces attributes
        for root element, conditionally with optional namespaces and
        prefix.
        """
    nmsp_dict: dict[str, str] = {}
    if self.namespaces:
        nmsp_dict = {f'xmlns{(p if p == '' else f':{p}')}': n for p, n in self.namespaces.items() if n != self.prefix_uri[1:-1]}
    return nmsp_dict