from __future__ import annotations
import decimal
import numbers
import sys
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
import pandas as pd
from pandas.api.extensions import (
from pandas.api.types import (
from pandas.core import arraylike
from pandas.core.algorithms import value_counts_internal as value_counts
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
from pandas.core.indexers import check_array_indexer
@register_extension_dtype
class DecimalDtype(ExtensionDtype):
    type = decimal.Decimal
    name = 'decimal'
    na_value = decimal.Decimal('NaN')
    _metadata = ('context',)

    def __init__(self, context=None) -> None:
        self.context = context or decimal.getcontext()

    def __repr__(self) -> str:
        return f'DecimalDtype(context={self.context})'

    @classmethod
    def construct_array_type(cls) -> type_t[DecimalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return DecimalArray

    @property
    def _is_numeric(self) -> bool:
        return True