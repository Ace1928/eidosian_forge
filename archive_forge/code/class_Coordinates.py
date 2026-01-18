from __future__ import annotations
from collections.abc import Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import (
import numpy as np
import pandas as pd
from xarray.core import formatting
from xarray.core.alignment import Aligner
from xarray.core.indexes import (
from xarray.core.merge import merge_coordinates_without_align, merge_coords
from xarray.core.types import DataVars, Self, T_DataArray, T_Xarray
from xarray.core.utils import (
from xarray.core.variable import Variable, as_variable, calculate_dimensions
class Coordinates(AbstractCoordinates):
    """Dictionary like container for Xarray coordinates (variables + indexes).

    This collection is a mapping of coordinate names to
    :py:class:`~xarray.DataArray` objects.

    It can be passed directly to the :py:class:`~xarray.Dataset` and
    :py:class:`~xarray.DataArray` constructors via their `coords` argument. This
    will add both the coordinates variables and their index.

    Coordinates are either:

    - returned via the :py:attr:`Dataset.coords` and :py:attr:`DataArray.coords`
      properties
    - built from Pandas or other index objects
      (e.g., :py:meth:`Coordinates.from_pandas_multiindex`)
    - built directly from coordinate data and Xarray ``Index`` objects (beware that
      no consistency check is done on those inputs)

    Parameters
    ----------
    coords: dict-like, optional
        Mapping where keys are coordinate names and values are objects that
        can be converted into a :py:class:`~xarray.Variable` object
        (see :py:func:`~xarray.as_variable`). If another
        :py:class:`~xarray.Coordinates` object is passed, its indexes
        will be added to the new created object.
    indexes: dict-like, optional
        Mapping where keys are coordinate names and values are
        :py:class:`~xarray.indexes.Index` objects. If None (default),
        pandas indexes will be created for each dimension coordinate.
        Passing an empty dictionary will skip this default behavior.

    Examples
    --------
    Create a dimension coordinate with a default (pandas) index:

    >>> xr.Coordinates({"x": [1, 2]})
    Coordinates:
      * x        (x) int64 16B 1 2

    Create a dimension coordinate with no index:

    >>> xr.Coordinates(coords={"x": [1, 2]}, indexes={})
    Coordinates:
        x        (x) int64 16B 1 2

    Create a new Coordinates object from existing dataset coordinates
    (indexes are passed):

    >>> ds = xr.Dataset(coords={"x": [1, 2]})
    >>> xr.Coordinates(ds.coords)
    Coordinates:
      * x        (x) int64 16B 1 2

    Create indexed coordinates from a ``pandas.MultiIndex`` object:

    >>> midx = pd.MultiIndex.from_product([["a", "b"], [0, 1]])
    >>> xr.Coordinates.from_pandas_multiindex(midx, "x")
    Coordinates:
      * x          (x) object 32B MultiIndex
      * x_level_0  (x) object 32B 'a' 'a' 'b' 'b'
      * x_level_1  (x) int64 32B 0 1 0 1

    Create a new Dataset object by passing a Coordinates object:

    >>> midx_coords = xr.Coordinates.from_pandas_multiindex(midx, "x")
    >>> xr.Dataset(coords=midx_coords)
    <xarray.Dataset> Size: 96B
    Dimensions:    (x: 4)
    Coordinates:
      * x          (x) object 32B MultiIndex
      * x_level_0  (x) object 32B 'a' 'a' 'b' 'b'
      * x_level_1  (x) int64 32B 0 1 0 1
    Data variables:
        *empty*

    """
    _data: DataWithCoords
    __slots__ = ('_data',)

    def __init__(self, coords: Mapping[Any, Any] | None=None, indexes: Mapping[Any, Index] | None=None) -> None:
        from xarray.core.dataset import Dataset
        if coords is None:
            coords = {}
        variables: dict[Hashable, Variable]
        default_indexes: dict[Hashable, PandasIndex] = {}
        coords_obj_indexes: dict[Hashable, Index] = {}
        if isinstance(coords, Coordinates):
            if indexes is not None:
                raise ValueError('passing both a ``Coordinates`` object and a mapping of indexes to ``Coordinates.__init__`` is not allowed (this constructor does not support merging them)')
            variables = {k: v.copy() for k, v in coords.variables.items()}
            coords_obj_indexes = dict(coords.xindexes)
        else:
            variables = {}
            for name, data in coords.items():
                var = as_variable(data, name=name, auto_convert=False)
                if var.dims == (name,) and indexes is None:
                    index, index_vars = create_default_index_implicit(var, list(coords))
                    default_indexes.update({k: index for k in index_vars})
                    variables.update(index_vars)
                else:
                    variables[name] = var
        if indexes is None:
            indexes = {}
        else:
            indexes = dict(indexes)
        indexes.update(default_indexes)
        indexes.update(coords_obj_indexes)
        no_coord_index = set(indexes) - set(variables)
        if no_coord_index:
            raise ValueError(f'no coordinate variables found for these indexes: {no_coord_index}')
        for k, idx in indexes.items():
            if not isinstance(idx, Index):
                raise TypeError(f"'{k}' is not an `xarray.indexes.Index` object")
        for k, v in variables.items():
            if k not in indexes:
                variables[k] = v.to_base_variable()
        self._data = Dataset._construct_direct(coord_names=set(variables), variables=variables, indexes=indexes)

    @classmethod
    def _construct_direct(cls, coords: dict[Any, Variable], indexes: dict[Any, Index], dims: dict[Any, int] | None=None) -> Self:
        from xarray.core.dataset import Dataset
        obj = object.__new__(cls)
        obj._data = Dataset._construct_direct(coord_names=set(coords), variables=coords, indexes=indexes, dims=dims)
        return obj

    @classmethod
    def from_pandas_multiindex(cls, midx: pd.MultiIndex, dim: str) -> Self:
        """Wrap a pandas multi-index as Xarray coordinates (dimension + levels).

        The returned coordinates can be directly assigned to a
        :py:class:`~xarray.Dataset` or :py:class:`~xarray.DataArray` via the
        ``coords`` argument of their constructor.

        Parameters
        ----------
        midx : :py:class:`pandas.MultiIndex`
            Pandas multi-index object.
        dim : str
            Dimension name.

        Returns
        -------
        coords : Coordinates
            A collection of Xarray indexed coordinates created from the multi-index.

        """
        xr_idx = PandasMultiIndex(midx, dim)
        variables = xr_idx.create_variables()
        indexes = {k: xr_idx for k in variables}
        return cls(coords=variables, indexes=indexes)

    @property
    def _names(self) -> set[Hashable]:
        return self._data._coord_names

    @property
    def dims(self) -> Frozen[Hashable, int] | tuple[Hashable, ...]:
        """Mapping from dimension names to lengths or tuple of dimension names."""
        return self._data.dims

    @property
    def sizes(self) -> Frozen[Hashable, int]:
        """Mapping from dimension names to lengths."""
        return self._data.sizes

    @property
    def dtypes(self) -> Frozen[Hashable, np.dtype]:
        """Mapping from coordinate names to dtypes.

        Cannot be modified directly.

        See Also
        --------
        Dataset.dtypes
        """
        return Frozen({n: v.dtype for n, v in self._data.variables.items()})

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to Coordinates contents as dict of Variable objects.

        This dictionary is frozen to prevent mutation.
        """
        return self._data.variables

    def to_dataset(self) -> Dataset:
        """Convert these coordinates into a new Dataset."""
        names = [name for name in self._data._variables if name in self._names]
        return self._data._copy_listed(names)

    def __getitem__(self, key: Hashable) -> DataArray:
        return self._data[key]

    def __delitem__(self, key: Hashable) -> None:
        del self._data.coords[key]

    def equals(self, other: Self) -> bool:
        """Two Coordinates objects are equal if they have matching variables,
        all of which are equal.

        See Also
        --------
        Coordinates.identical
        """
        if not isinstance(other, Coordinates):
            return False
        return self.to_dataset().equals(other.to_dataset())

    def identical(self, other: Self) -> bool:
        """Like equals, but also checks all variable attributes.

        See Also
        --------
        Coordinates.equals
        """
        if not isinstance(other, Coordinates):
            return False
        return self.to_dataset().identical(other.to_dataset())

    def _update_coords(self, coords: dict[Hashable, Variable], indexes: Mapping[Any, Index]) -> None:
        self._data.coords._update_coords(coords, indexes)

    def _drop_coords(self, coord_names):
        self._data.coords._drop_coords(coord_names)

    def _merge_raw(self, other, reflexive):
        """For use with binary arithmetic."""
        if other is None:
            variables = dict(self.variables)
            indexes = dict(self.xindexes)
        else:
            coord_list = [self, other] if not reflexive else [other, self]
            variables, indexes = merge_coordinates_without_align(coord_list)
        return (variables, indexes)

    @contextmanager
    def _merge_inplace(self, other):
        """For use with in-place binary arithmetic."""
        if other is None:
            yield
        else:
            prioritized = {k: (v, None) for k, v in self.variables.items() if k not in self.xindexes}
            variables, indexes = merge_coordinates_without_align([self, other], prioritized)
            yield
            self._update_coords(variables, indexes)

    def merge(self, other: Mapping[Any, Any] | None) -> Dataset:
        """Merge two sets of coordinates to create a new Dataset

        The method implements the logic used for joining coordinates in the
        result of a binary operation performed on xarray objects:

        - If two index coordinates conflict (are not equal), an exception is
          raised. You must align your data before passing it to this method.
        - If an index coordinate and a non-index coordinate conflict, the non-
          index coordinate is dropped.
        - If two non-index coordinates conflict, both are dropped.

        Parameters
        ----------
        other : dict-like, optional
            A :py:class:`Coordinates` object or any mapping that can be turned
            into coordinates.

        Returns
        -------
        merged : Dataset
            A new Dataset with merged coordinates.
        """
        from xarray.core.dataset import Dataset
        if other is None:
            return self.to_dataset()
        if not isinstance(other, Coordinates):
            other = Dataset(coords=other).coords
        coords, indexes = merge_coordinates_without_align([self, other])
        coord_names = set(coords)
        return Dataset._construct_direct(variables=coords, coord_names=coord_names, indexes=indexes)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.update({key: value})

    def update(self, other: Mapping[Any, Any]) -> None:
        """Update this Coordinates variables with other coordinate variables."""
        if not len(other):
            return
        other_coords: Coordinates
        if isinstance(other, Coordinates):
            other_coords = other
        else:
            other_coords = create_coords_with_default_indexes(getattr(other, 'variables', other))
        coords_to_align = drop_indexed_coords(set(other_coords) & set(other), self)
        coords, indexes = merge_coords([coords_to_align, other_coords], priority_arg=1, indexes=coords_to_align.xindexes)
        self._drop_coords(self._names - coords_to_align._names)
        self._update_coords(coords, indexes)

    def assign(self, coords: Mapping | None=None, **coords_kwargs: Any) -> Self:
        """Assign new coordinates (and indexes) to a Coordinates object, returning
        a new object with all the original coordinates in addition to the new ones.

        Parameters
        ----------
        coords : mapping of dim to coord, optional
            A mapping whose keys are the names of the coordinates and values are the
            coordinates to assign. The mapping will generally be a dict or
            :class:`Coordinates`.

            * If a value is a standard data value — for example, a ``DataArray``,
              scalar, or array — the data is simply assigned as a coordinate.

            * A coordinate can also be defined and attached to an existing dimension
              using a tuple with the first element the dimension name and the second
              element the values for this new coordinate.

        **coords_kwargs
            The keyword arguments form of ``coords``.
            One of ``coords`` or ``coords_kwargs`` must be provided.

        Returns
        -------
        new_coords : Coordinates
            A new Coordinates object with the new coordinates (and indexes)
            in addition to all the existing coordinates.

        Examples
        --------
        >>> coords = xr.Coordinates()
        >>> coords
        Coordinates:
            *empty*

        >>> coords.assign(x=[1, 2])
        Coordinates:
          * x        (x) int64 16B 1 2

        >>> midx = pd.MultiIndex.from_product([["a", "b"], [0, 1]])
        >>> coords.assign(xr.Coordinates.from_pandas_multiindex(midx, "y"))
        Coordinates:
          * y          (y) object 32B MultiIndex
          * y_level_0  (y) object 32B 'a' 'a' 'b' 'b'
          * y_level_1  (y) int64 32B 0 1 0 1

        """
        coords = either_dict_or_kwargs(coords, coords_kwargs, 'assign')
        new_coords = self.copy()
        new_coords.update(coords)
        return new_coords

    def _overwrite_indexes(self, indexes: Mapping[Any, Index], variables: Mapping[Any, Variable] | None=None) -> Self:
        results = self.to_dataset()._overwrite_indexes(indexes, variables)
        return cast(Self, results.coords)

    def _reindex_callback(self, aligner: Aligner, dim_pos_indexers: dict[Hashable, Any], variables: dict[Hashable, Variable], indexes: dict[Hashable, Index], fill_value: Any, exclude_dims: frozenset[Hashable], exclude_vars: frozenset[Hashable]) -> Self:
        """Callback called from ``Aligner`` to create a new reindexed Coordinate."""
        aligned = self.to_dataset()._reindex_callback(aligner, dim_pos_indexers, variables, indexes, fill_value, exclude_dims, exclude_vars)
        return cast(Self, aligned.coords)

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython."""
        return self._data._ipython_key_completions_()

    def copy(self, deep: bool=False, memo: dict[int, Any] | None=None) -> Self:
        """Return a copy of this Coordinates object."""
        variables = {k: v._copy(deep=deep, memo=memo) for k, v in self.variables.items()}
        return cast(Self, Coordinates._construct_direct(coords=variables, indexes=dict(self.xindexes), dims=dict(self.sizes)))