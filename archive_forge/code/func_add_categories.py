from __future__ import annotations
from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import (
from pandas.core.algorithms import (
from pandas.core.arrays._mixins import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.io.formats import console
def add_categories(self, new_categories) -> Self:
    """
        Add new categories.

        `new_categories` will be included at the last/highest place in the
        categories and will be unused directly after this call.

        Parameters
        ----------
        new_categories : category or list-like of category
           The new categories to be included.

        Returns
        -------
        Categorical
            Categorical with new categories added.

        Raises
        ------
        ValueError
            If the new categories include old categories or do not validate as
            categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['c', 'b', 'c'])
        >>> c
        ['c', 'b', 'c']
        Categories (2, object): ['b', 'c']

        >>> c.add_categories(['d', 'a'])
        ['c', 'b', 'c']
        Categories (4, object): ['b', 'c', 'd', 'a']
        """
    if not is_list_like(new_categories):
        new_categories = [new_categories]
    already_included = set(new_categories) & set(self.dtype.categories)
    if len(already_included) != 0:
        raise ValueError(f'new categories must not include old categories: {already_included}')
    if hasattr(new_categories, 'dtype'):
        from pandas import Series
        dtype = find_common_type([self.dtype.categories.dtype, new_categories.dtype])
        new_categories = Series(list(self.dtype.categories) + list(new_categories), dtype=dtype)
    else:
        new_categories = list(self.dtype.categories) + list(new_categories)
    new_dtype = CategoricalDtype(new_categories, self.ordered)
    cat = self.copy()
    codes = coerce_indexer_dtype(cat._ndarray, new_dtype.categories)
    NDArrayBacked.__init__(cat, codes, new_dtype)
    return cat