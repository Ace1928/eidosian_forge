from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
class ModinDtypes:
    """
    A class that hides the various implementations of the dtypes needed for optimization.

    Parameters
    ----------
    value : pandas.Series, callable, DtypesDescriptor or ModinDtypes, optional
    """

    def __init__(self, value: Optional[Union[Callable, pandas.Series, DtypesDescriptor, 'ModinDtypes']]):
        if callable(value) or isinstance(value, pandas.Series):
            self._value = value
        elif isinstance(value, DtypesDescriptor):
            self._value = value.to_series() if value.is_materialized else value
        elif isinstance(value, type(self)):
            self._value = value.copy()._value
        elif isinstance(value, None):
            self._value = DtypesDescriptor()
        else:
            raise ValueError(f"ModinDtypes doesn't work with '{value}'")

    def __repr__(self):
        return f'ModinDtypes:\n\tvalue type: {type(self._value)};\n\tvalue:\n\t{self._value}'

    def __str__(self):
        return self.__repr__()

    @property
    def is_materialized(self) -> bool:
        """
        Check if the internal representation is materialized.

        Returns
        -------
        bool
        """
        return isinstance(self._value, pandas.Series)

    def get_dtypes_set(self) -> set[np.dtype]:
        """
        Get a set of dtypes from the descriptor.

        Returns
        -------
        set[np.dtype]
        """
        if isinstance(self._value, DtypesDescriptor):
            return self._value.get_dtypes_set()
        if not self.is_materialized:
            self.get()
        return set(self._value.values)

    def maybe_specify_new_frame_ref(self, new_parent: 'PandasDataframe') -> 'ModinDtypes':
        """
        Set a new parent for the stored value if needed.

        Parameters
        ----------
        new_parent : PandasDataframe

        Returns
        -------
        ModinDtypes
            A copy of ``ModinDtypes`` with updated parent.
        """
        new_self = self.copy()
        if new_self.is_materialized:
            LazyProxyCategoricalDtype.update_dtypes(new_self._value, new_parent)
            return new_self
        if isinstance(self._value, DtypesDescriptor):
            new_self._value.update_parent(new_parent)
            return new_self
        return new_self

    def lazy_get(self, ids: list, numeric_index: bool=False) -> 'ModinDtypes':
        """
        Get new ``ModinDtypes`` for a subset of columns without triggering any computations.

        Parameters
        ----------
        ids : list of index labels or positional indexers
            Columns for the subset.
        numeric_index : bool, default: False
            Whether `ids` are positional indixes or column labels to take.

        Returns
        -------
        ModinDtypes
            ``ModinDtypes`` that describes dtypes for columns specified in `ids`.
        """
        if isinstance(self._value, DtypesDescriptor):
            res = self._value.lazy_get(ids, numeric_index)
            return ModinDtypes(res)
        elif callable(self._value):
            new_self = self.copy()
            old_value = new_self._value
            new_self._value = lambda: old_value().iloc[ids] if numeric_index else old_value()[ids]
            return new_self
        ErrorMessage.catch_bugs_and_request_email(failure_condition=not self.is_materialized)
        return ModinDtypes(self._value.iloc[ids] if numeric_index else self._value[ids])

    @classmethod
    def concat(cls, values: list, axis: int=0) -> 'ModinDtypes':
        """
        Concatenate dtypes.

        Parameters
        ----------
        values : list of DtypesDescriptors, pandas.Series, ModinDtypes and Nones
        axis : int, default: 0
            If ``axis == 0``: concatenate column names. This implements the logic of
            how dtypes are combined on ``pd.concat([df1, df2], axis=1)``.
            If ``axis == 1``: perform a union join for the column names described by
            `values` and then find common dtypes for the columns appeared to be in
            an intersection. This implements the logic of how dtypes are combined on
            ``pd.concat([df1, df2], axis=0).dtypes``.

        Returns
        -------
        ModinDtypes
        """
        preprocessed_vals = []
        for val in values:
            if isinstance(val, cls):
                val = val._value
            if isinstance(val, (DtypesDescriptor, pandas.Series)) or val is None:
                preprocessed_vals.append(val)
            else:
                raise NotImplementedError(type(val))
        try:
            desc = DtypesDescriptor.concat(preprocessed_vals, axis=axis)
        except NotImplementedError as e:
            if axis == 0 and 'duplicated' not in e.args[0].lower() or not all((isinstance(val, pandas.Series) for val in values)):
                raise e
            desc = pandas.concat(values)
        return ModinDtypes(desc)

    def set_index(self, new_index: Union[pandas.Index, 'ModinIndex']) -> 'ModinDtypes':
        """
        Set new column names for stored dtypes.

        Parameters
        ----------
        new_index : pandas.Index or ModinIndex

        Returns
        -------
        ModinDtypes
            New ``ModinDtypes`` with updated column names.
        """
        new_self = self.copy()
        if self.is_materialized:
            new_self._value.index = new_index
            return new_self
        elif callable(self._value):
            old_val = new_self._value
            new_self._value = lambda: old_val().set_axis(new_index)
            return new_self
        elif isinstance(new_self._value, DtypesDescriptor):
            new_self._value = new_self._value.set_index(new_index)
            return new_self
        else:
            raise NotImplementedError()

    def get(self) -> pandas.Series:
        """
        Get the materialized internal representation.

        Returns
        -------
        pandas.Series
        """
        if not self.is_materialized:
            if callable(self._value):
                self._value = self._value()
                if self._value is None:
                    self._value = pandas.Series([])
            elif isinstance(self._value, DtypesDescriptor):
                self._value = self._value.to_series()
            else:
                raise NotImplementedError(type(self._value))
        return self._value

    def __len__(self):
        """
        Redirect the 'len' request to the internal representation.

        Returns
        -------
        int

        Notes
        -----
        Executing this function materializes the data.
        """
        if not self.is_materialized:
            self.get()
        return len(self._value)

    def __reduce__(self):
        """
        Serialize an object of this class.

        Returns
        -------
        tuple

        Notes
        -----
        The default implementation generates a recursion error. In a short:
        during the construction of the object, `__getattr__` function is called, which
        is not intended to be used in situations where the object is not initialized.
        """
        return (self.__class__, (self._value,))

    def __getattr__(self, name):
        """
        Redirect access to non-existent attributes to the internal representation.

        This is necessary so that objects of this class in most cases mimic the behavior
        of the ``pandas.Series``. The main limitations of the current approach are type
        checking and the use of this object where pandas dtypes are supposed to be used.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        object
            Attribute.

        Notes
        -----
        Executing this function materializes the data.
        """
        if not self.is_materialized:
            self.get()
        return self._value.__getattribute__(name)

    def copy(self) -> 'ModinDtypes':
        """
        Copy an object without materializing the internal representation.

        Returns
        -------
        ModinDtypes
        """
        idx_cache = self._value
        if not callable(idx_cache):
            idx_cache = idx_cache.copy()
        return ModinDtypes(idx_cache)

    def __getitem__(self, key):
        if not self.is_materialized:
            self.get()
        return self._value.__getitem__(key)

    def __setitem__(self, key, item):
        if not self.is_materialized:
            self.get()
        self._value.__setitem__(key, item)

    def __iter__(self):
        if not self.is_materialized:
            self.get()
        return iter(self._value)

    def __contains__(self, key):
        if not self.is_materialized:
            self.get()
        return key in self._value