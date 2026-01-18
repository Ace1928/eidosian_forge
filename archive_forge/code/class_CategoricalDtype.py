from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
@register_extension_dtype
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    """
    Type for categorical data with the categories and orderedness.

    Parameters
    ----------
    categories : sequence, optional
        Must be unique, and must not contain any nulls.
        The categories are stored in an Index,
        and if an index is provided the dtype of that index will be used.
    ordered : bool or None, default False
        Whether or not this categorical is treated as a ordered categorical.
        None can be used to maintain the ordered value of existing categoricals when
        used in operations that combine categoricals, e.g. astype, and will resolve to
        False if there is no existing ordered to maintain.

    Attributes
    ----------
    categories
    ordered

    Methods
    -------
    None

    See Also
    --------
    Categorical : Represent a categorical variable in classic R / S-plus fashion.

    Notes
    -----
    This class is useful for specifying the type of a ``Categorical``
    independent of the values. See :ref:`categorical.categoricaldtype`
    for more.

    Examples
    --------
    >>> t = pd.CategoricalDtype(categories=['b', 'a'], ordered=True)
    >>> pd.Series(['a', 'b', 'a', 'c'], dtype=t)
    0      a
    1      b
    2      a
    3    NaN
    dtype: category
    Categories (2, object): ['b' < 'a']

    An empty CategoricalDtype with a specific dtype can be created
    by providing an empty index. As follows,

    >>> pd.CategoricalDtype(pd.DatetimeIndex([])).categories.dtype
    dtype('<M8[ns]')
    """
    name = 'category'
    type: type[CategoricalDtypeType] = CategoricalDtypeType
    kind: str_type = 'O'
    str = '|O08'
    base = np.dtype('O')
    _metadata = ('categories', 'ordered')
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _supports_2d = False
    _can_fast_transpose = False

    def __init__(self, categories=None, ordered: Ordered=False) -> None:
        self._finalize(categories, ordered, fastpath=False)

    @classmethod
    def _from_fastpath(cls, categories=None, ordered: bool | None=None) -> CategoricalDtype:
        self = cls.__new__(cls)
        self._finalize(categories, ordered, fastpath=True)
        return self

    @classmethod
    def _from_categorical_dtype(cls, dtype: CategoricalDtype, categories=None, ordered: Ordered | None=None) -> CategoricalDtype:
        if categories is ordered is None:
            return dtype
        if categories is None:
            categories = dtype.categories
        if ordered is None:
            ordered = dtype.ordered
        return cls(categories, ordered)

    @classmethod
    def _from_values_or_dtype(cls, values=None, categories=None, ordered: bool | None=None, dtype: Dtype | None=None) -> CategoricalDtype:
        """
        Construct dtype from the input parameters used in :class:`Categorical`.

        This constructor method specifically does not do the factorization
        step, if that is needed to find the categories. This constructor may
        therefore return ``CategoricalDtype(categories=None, ordered=None)``,
        which may not be useful. Additional steps may therefore have to be
        taken to create the final dtype.

        The return dtype is specified from the inputs in this prioritized
        order:
        1. if dtype is a CategoricalDtype, return dtype
        2. if dtype is the string 'category', create a CategoricalDtype from
           the supplied categories and ordered parameters, and return that.
        3. if values is a categorical, use value.dtype, but override it with
           categories and ordered if either/both of those are not None.
        4. if dtype is None and values is not a categorical, construct the
           dtype from categories and ordered, even if either of those is None.

        Parameters
        ----------
        values : list-like, optional
            The list-like must be 1-dimensional.
        categories : list-like, optional
            Categories for the CategoricalDtype.
        ordered : bool, optional
            Designating if the categories are ordered.
        dtype : CategoricalDtype or the string "category", optional
            If ``CategoricalDtype``, cannot be used together with
            `categories` or `ordered`.

        Returns
        -------
        CategoricalDtype

        Examples
        --------
        >>> pd.CategoricalDtype._from_values_or_dtype()
        CategoricalDtype(categories=None, ordered=None, categories_dtype=None)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     categories=['a', 'b'], ordered=True
        ... )
        CategoricalDtype(categories=['a', 'b'], ordered=True, categories_dtype=object)
        >>> dtype1 = pd.CategoricalDtype(['a', 'b'], ordered=True)
        >>> dtype2 = pd.CategoricalDtype(['x', 'y'], ordered=False)
        >>> c = pd.Categorical([0, 1], dtype=dtype1)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     c, ['x', 'y'], ordered=True, dtype=dtype2
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Cannot specify `categories` or `ordered` together with
        `dtype`.

        The supplied dtype takes precedence over values' dtype:

        >>> pd.CategoricalDtype._from_values_or_dtype(c, dtype=dtype2)
        CategoricalDtype(categories=['x', 'y'], ordered=False, categories_dtype=object)
        """
        if dtype is not None:
            if isinstance(dtype, str):
                if dtype == 'category':
                    if ordered is None and cls.is_dtype(values):
                        ordered = values.dtype.ordered
                    dtype = CategoricalDtype(categories, ordered)
                else:
                    raise ValueError(f'Unknown dtype {repr(dtype)}')
            elif categories is not None or ordered is not None:
                raise ValueError('Cannot specify `categories` or `ordered` together with `dtype`.')
            elif not isinstance(dtype, CategoricalDtype):
                raise ValueError(f'Cannot not construct CategoricalDtype from {dtype}')
        elif cls.is_dtype(values):
            dtype = values.dtype._from_categorical_dtype(values.dtype, categories, ordered)
        else:
            dtype = CategoricalDtype(categories, ordered)
        return cast(CategoricalDtype, dtype)

    @classmethod
    def construct_from_string(cls, string: str_type) -> CategoricalDtype:
        """
        Construct a CategoricalDtype from a string.

        Parameters
        ----------
        string : str
            Must be the string "category" in order to be successfully constructed.

        Returns
        -------
        CategoricalDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a CategoricalDtype cannot be constructed from the input.
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string != cls.name:
            raise TypeError(f"Cannot construct a 'CategoricalDtype' from '{string}'")
        return cls(ordered=None)

    def _finalize(self, categories, ordered: Ordered, fastpath: bool=False) -> None:
        if ordered is not None:
            self.validate_ordered(ordered)
        if categories is not None:
            categories = self.validate_categories(categories, fastpath=fastpath)
        self._categories = categories
        self._ordered = ordered

    def __setstate__(self, state: MutableMapping[str_type, Any]) -> None:
        self._categories = state.pop('categories', None)
        self._ordered = state.pop('ordered', False)

    def __hash__(self) -> int:
        if self.categories is None:
            if self.ordered:
                return -1
            else:
                return -2
        return int(self._hash_categories)

    def __eq__(self, other: object) -> bool:
        """
        Rules for CDT equality:
        1) Any CDT is equal to the string 'category'
        2) Any CDT is equal to itself
        3) Any CDT is equal to a CDT with categories=None regardless of ordered
        4) A CDT with ordered=True is only equal to another CDT with
           ordered=True and identical categories in the same order
        5) A CDT with ordered={False, None} is only equal to another CDT with
           ordered={False, None} and identical categories, but same order is
           not required. There is no distinction between False/None.
        6) Any other comparison returns False
        """
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not (hasattr(other, 'ordered') and hasattr(other, 'categories')):
            return False
        elif self.categories is None or other.categories is None:
            return self.categories is other.categories
        elif self.ordered or other.ordered:
            return self.ordered == other.ordered and self.categories.equals(other.categories)
        else:
            left = self.categories
            right = other.categories
            if not left.dtype == right.dtype:
                return False
            if len(left) != len(right):
                return False
            if self.categories.equals(other.categories):
                return True
            if left.dtype != object:
                indexer = left.get_indexer(right)
                return (indexer != -1).all()
            return set(left) == set(right)

    def __repr__(self) -> str_type:
        if self.categories is None:
            data = 'None'
            dtype = 'None'
        else:
            data = self.categories._format_data(name=type(self).__name__)
            if isinstance(self.categories, ABCRangeIndex):
                data = str(self.categories._range)
            data = data.rstrip(', ')
            dtype = self.categories.dtype
        return f'CategoricalDtype(categories={data}, ordered={self.ordered}, categories_dtype={dtype})'

    @cache_readonly
    def _hash_categories(self) -> int:
        from pandas.core.util.hashing import combine_hash_arrays, hash_array, hash_tuples
        categories = self.categories
        ordered = self.ordered
        if len(categories) and isinstance(categories[0], tuple):
            cat_list = list(categories)
            cat_array = hash_tuples(cat_list)
        else:
            if categories.dtype == 'O' and len({type(x) for x in categories}) != 1:
                hashed = hash((tuple(categories), ordered))
                return hashed
            if DatetimeTZDtype.is_dtype(categories.dtype):
                categories = categories.view('datetime64[ns]')
            cat_array = hash_array(np.asarray(categories), categorize=False)
        if ordered:
            cat_array = np.vstack([cat_array, np.arange(len(cat_array), dtype=cat_array.dtype)])
        else:
            cat_array = np.array([cat_array])
        combined_hashed = combine_hash_arrays(iter(cat_array), num_items=len(cat_array))
        return np.bitwise_xor.reduce(combined_hashed)

    @classmethod
    def construct_array_type(cls) -> type_t[Categorical]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas import Categorical
        return Categorical

    @staticmethod
    def validate_ordered(ordered: Ordered) -> None:
        """
        Validates that we have a valid ordered parameter. If
        it is not a boolean, a TypeError will be raised.

        Parameters
        ----------
        ordered : object
            The parameter to be verified.

        Raises
        ------
        TypeError
            If 'ordered' is not a boolean.
        """
        if not is_bool(ordered):
            raise TypeError("'ordered' must either be 'True' or 'False'")

    @staticmethod
    def validate_categories(categories, fastpath: bool=False) -> Index:
        """
        Validates that we have good categories

        Parameters
        ----------
        categories : array-like
        fastpath : bool
            Whether to skip nan and uniqueness checks

        Returns
        -------
        categories : Index
        """
        from pandas.core.indexes.base import Index
        if not fastpath and (not is_list_like(categories)):
            raise TypeError(f"Parameter 'categories' must be list-like, was {repr(categories)}")
        if not isinstance(categories, ABCIndex):
            categories = Index._with_infer(categories, tupleize_cols=False)
        if not fastpath:
            if categories.hasnans:
                raise ValueError('Categorical categories cannot be null')
            if not categories.is_unique:
                raise ValueError('Categorical categories must be unique')
        if isinstance(categories, ABCCategoricalIndex):
            categories = categories.categories
        return categories

    def update_dtype(self, dtype: str_type | CategoricalDtype) -> CategoricalDtype:
        """
        Returns a CategoricalDtype with categories and ordered taken from dtype
        if specified, otherwise falling back to self if unspecified

        Parameters
        ----------
        dtype : CategoricalDtype

        Returns
        -------
        new_dtype : CategoricalDtype
        """
        if isinstance(dtype, str) and dtype == 'category':
            return self
        elif not self.is_dtype(dtype):
            raise ValueError(f'a CategoricalDtype must be passed to perform an update, got {repr(dtype)}')
        else:
            dtype = cast(CategoricalDtype, dtype)
        new_categories = dtype.categories if dtype.categories is not None else self.categories
        new_ordered = dtype.ordered if dtype.ordered is not None else self.ordered
        return CategoricalDtype(new_categories, new_ordered)

    @property
    def categories(self) -> Index:
        """
        An ``Index`` containing the unique categories allowed.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=True)
        >>> cat_type.categories
        Index(['a', 'b'], dtype='object')
        """
        return self._categories

    @property
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=True)
        >>> cat_type.ordered
        True

        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=False)
        >>> cat_type.ordered
        False
        """
        return self._ordered

    @property
    def _is_boolean(self) -> bool:
        from pandas.core.dtypes.common import is_bool_dtype
        return is_bool_dtype(self.categories)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        if all((isinstance(x, CategoricalDtype) for x in dtypes)):
            first = dtypes[0]
            if all((first == other for other in dtypes[1:])):
                return first
        non_init_cats = [isinstance(x, CategoricalDtype) and x.categories is None for x in dtypes]
        if all(non_init_cats):
            return self
        elif any(non_init_cats):
            return None
        dtypes = [x.subtype if isinstance(x, SparseDtype) else x for x in dtypes]
        non_cat_dtypes = [x.categories.dtype if isinstance(x, CategoricalDtype) else x for x in dtypes]
        from pandas.core.dtypes.cast import find_common_type
        return find_common_type(non_cat_dtypes)

    @cache_readonly
    def index_class(self) -> type_t[CategoricalIndex]:
        from pandas import CategoricalIndex
        return CategoricalIndex