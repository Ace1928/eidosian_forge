from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np
import pandas
from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings
@_inherit_docstrings(pandas.core.arrays.arrow.StructAccessor)
class StructAccessor(ClassLogger):
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series=None):
        self._series = data
        self._query_compiler = data._query_compiler

    @pandas.util.cache_readonly
    def _Series(self):
        from modin.pandas.series import Series
        return Series

    @property
    def dtypes(self):
        return self._Series(query_compiler=self._query_compiler.struct_dtypes())

    def field(self, name_or_index):
        return self._Series(query_compiler=self._query_compiler.struct_field(name_or_index=name_or_index))

    def explode(self):
        from modin.pandas.dataframe import DataFrame
        return DataFrame(query_compiler=self._query_compiler.struct_explode())