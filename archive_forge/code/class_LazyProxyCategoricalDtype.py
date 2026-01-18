from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
class LazyProxyCategoricalDtype(pandas.CategoricalDtype):
    """
    A lazy proxy representing ``pandas.CategoricalDtype``.

    Parameters
    ----------
    categories : list-like, optional
    ordered : bool, default: False

    Notes
    -----
    Important note! One shouldn't use the class' constructor to instantiate a proxy instance,
    it's intended only for compatibility purposes! In order to create a new proxy instance
    use the appropriate class method `._build_proxy(...)`.
    """

    def __init__(self, categories=None, ordered=False):
        self._parent, self._column_name, self._categories_val, self._materializer = (None, None, None, None)
        super().__init__(categories, ordered)

    @staticmethod
    def update_dtypes(dtypes, new_parent):
        """
        Update a parent for categorical proxies in a dtype object.

        Parameters
        ----------
        dtypes : dict-like
            A dict-like object describing dtypes. The method will walk through every dtype
            an update parents for categorical proxies inplace.
        new_parent : object
        """
        for key, value in dtypes.items():
            if isinstance(value, LazyProxyCategoricalDtype):
                dtypes[key] = value._update_proxy(new_parent, column_name=key)

    def _update_proxy(self, parent, column_name):
        """
        Create a new proxy, if either parent or column name are different.

        Parameters
        ----------
        parent : object
            Source object to extract categories on demand.
        column_name : str
            Column name of the categorical column in the source object.

        Returns
        -------
        pandas.CategoricalDtype or LazyProxyCategoricalDtype
        """
        if self._is_materialized:
            return pandas.CategoricalDtype(self.categories, ordered=self._ordered)
        elif parent is self._parent and column_name == self._column_name:
            return self
        else:
            return self._build_proxy(parent, column_name, self._materializer)

    @classmethod
    def _build_proxy(cls, parent, column_name, materializer, dtype=None):
        """
        Construct a lazy proxy.

        Parameters
        ----------
        parent : object
            Source object to extract categories on demand.
        column_name : str
            Column name of the categorical column in the source object.
        materializer : callable(parent, column_name) -> pandas.CategoricalDtype
            A function to call in order to extract categorical values.
        dtype : dtype, optional
            The categories dtype.

        Returns
        -------
        LazyProxyCategoricalDtype
        """
        result = cls()
        result._parent = parent
        result._column_name = column_name
        result._materializer = materializer
        result._dtype = dtype
        return result

    def _get_dtype(self):
        """
        Get the categories dtype.

        Returns
        -------
        dtype
        """
        if self._dtype is None:
            self._dtype = self.categories.dtype
        return self._dtype

    def __reduce__(self):
        """
        Serialize an object of this class.

        Returns
        -------
        tuple

        Notes
        -----
        This object is serialized into a ``pandas.CategoricalDtype`` as an actual proxy can't be
        properly serialized because of the references it stores for its potentially distributed parent.
        """
        return (pandas.CategoricalDtype, (self.categories, self.ordered))

    @property
    def _categories(self):
        """
        Get materialized categorical values.

        Returns
        -------
        pandas.Index
        """
        if not self._is_materialized:
            self._materialize_categories()
        return self._categories_val

    @_categories.setter
    def _categories(self, categories):
        """
        Set new categorical values.

        Parameters
        ----------
        categories : list-like
        """
        self._categories_val = categories
        self._parent = None
        self._materializer = None
        self._dtype = None

    @property
    def _is_materialized(self) -> bool:
        """
        Check whether categorical values were already materialized.

        Returns
        -------
        bool
        """
        return self._categories_val is not None

    def _materialize_categories(self):
        """Materialize actual categorical values."""
        ErrorMessage.catch_bugs_and_request_email(failure_condition=self._parent is None, extra_log="attempted to materialize categories with parent being 'None'")
        categoricals = self._materializer(self._parent, self._column_name)
        self._categories = categoricals.categories
        self._ordered = categoricals.ordered