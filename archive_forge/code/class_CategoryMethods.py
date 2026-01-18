from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np
import pandas
from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings
@_inherit_docstrings(pandas.core.arrays.categorical.CategoricalAccessor)
class CategoryMethods(ClassLogger):
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series):
        self._series = data
        self._query_compiler = data._query_compiler

    @pandas.util.cache_readonly
    def _Series(self):
        from modin.pandas.series import Series
        return Series

    @property
    def categories(self):
        return self._series.dtype.categories

    @categories.setter
    def categories(self, categories):

        def set_categories(series, categories):
            series.cat.categories = categories
        self._series._default_to_pandas(set_categories, categories=categories)

    @property
    def ordered(self):
        return self._series.dtype.ordered

    @property
    def codes(self):
        return self._Series(query_compiler=self._query_compiler.cat_codes())

    def rename_categories(self, new_categories):
        return self._default_to_pandas(pandas.Series.cat.rename_categories, new_categories)

    def reorder_categories(self, new_categories, ordered=None):
        return self._default_to_pandas(pandas.Series.cat.reorder_categories, new_categories, ordered=ordered)

    def add_categories(self, new_categories):
        return self._default_to_pandas(pandas.Series.cat.add_categories, new_categories)

    def remove_categories(self, removals):
        return self._default_to_pandas(pandas.Series.cat.remove_categories, removals)

    def remove_unused_categories(self):
        return self._default_to_pandas(pandas.Series.cat.remove_unused_categories)

    def set_categories(self, new_categories, ordered=None, rename=False):
        return self._default_to_pandas(pandas.Series.cat.set_categories, new_categories, ordered=ordered, rename=rename)

    def as_ordered(self):
        return self._default_to_pandas(pandas.Series.cat.as_ordered)

    def as_unordered(self):
        return self._default_to_pandas(pandas.Series.cat.as_unordered)

    def _default_to_pandas(self, op, *args, **kwargs):
        """
        Convert `self` to pandas type and call a pandas cat.`op` on it.

        Parameters
        ----------
        op : str
            Name of pandas function.
        *args : list
            Additional positional arguments to be passed in `op`.
        **kwargs : dict
            Additional keywords arguments to be passed in `op`.

        Returns
        -------
        object
            Result of operation.
        """
        return self._series._default_to_pandas(lambda series: op(series.cat, *args, **kwargs))