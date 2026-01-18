from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray
from sympy.utilities.iterables import flatten
import functools
class MutableSparseNDimArray(MutableNDimArray, SparseNDimArray):

    def __new__(cls, iterable=None, shape=None, **kwargs):
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        self = object.__new__(cls)
        self._shape = shape
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x, y: x * y, shape) if shape else len(flat_list)
        if isinstance(flat_list, (dict, Dict)):
            self._sparse_array = dict(flat_list)
            return self
        self._sparse_array = {}
        for i, el in enumerate(flatten(flat_list)):
            if el != 0:
                self._sparse_array[i] = _sympify(el)
        return self

    def __setitem__(self, index, value):
        """Allows to set items to MutableDenseNDimArray.

        Examples
        ========

        >>> from sympy import MutableSparseNDimArray
        >>> a = MutableSparseNDimArray.zeros(2, 2)
        >>> a[0, 0] = 1
        >>> a[1, 1] = 1
        >>> a
        [[1, 0], [0, 1]]
        """
        if isinstance(index, tuple) and any((isinstance(i, slice) for i in index)):
            value, eindices, slice_offsets = self._get_slice_data_for_array_assignment(index, value)
            for i in eindices:
                other_i = [ind - j for ind, j in zip(i, slice_offsets) if j is not None]
                other_value = value[other_i]
                complete_index = self._parse_index(i)
                if other_value != 0:
                    self._sparse_array[complete_index] = other_value
                elif complete_index in self._sparse_array:
                    self._sparse_array.pop(complete_index)
        else:
            index = self._parse_index(index)
            value = _sympify(value)
            if value == 0 and index in self._sparse_array:
                self._sparse_array.pop(index)
            else:
                self._sparse_array[index] = value

    def as_immutable(self):
        return ImmutableSparseNDimArray(self)

    @property
    def free_symbols(self):
        return {i for j in self._sparse_array.values() for i in j.free_symbols}