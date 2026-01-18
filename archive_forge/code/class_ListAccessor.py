from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np
import pandas
from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings
@_inherit_docstrings(pandas.core.arrays.arrow.ListAccessor)
class ListAccessor(ClassLogger):
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series=None):
        self._series = data
        self._query_compiler = data._query_compiler

    @pandas.util.cache_readonly
    def _Series(self):
        from .series import Series
        return Series

    def flatten(self):
        return self._Series(query_compiler=self._query_compiler.list_flatten())

    def len(self):
        return self._Series(query_compiler=self._query_compiler.list_len())

    def __getitem__(self, key):
        return self._Series(query_compiler=self._query_compiler.list__getitem__(key=key))