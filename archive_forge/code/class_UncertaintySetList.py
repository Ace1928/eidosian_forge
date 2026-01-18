import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
class UncertaintySetList(MutableSequence):
    """
    Wrapper around a list of uncertainty sets, all of which have
    an immutable common dimension.

    Parameters
    ----------
    uncertainty_sets : iterable, optional
        Sequence of uncertainty sets.
    name : str or None, optional
        Name of the uncertainty set list.
    min_length : int or None, optional
        Minimum required length of the sequence. If `None` is
        provided, then the minimum required length is set to 0.
    """

    def __init__(self, uncertainty_sets=[], name=None, min_length=None):
        """Initialize self (see class docstring)."""
        self._name = name
        self._min_length = 0 if min_length is None else min_length
        initlist = list(uncertainty_sets)
        if len(initlist) < self._min_length:
            raise ValueError(f'Attempting to initialize uncertainty set list {self._name!r} of minimum required length {self._min_length} with an iterable of length {len(initlist)}')
        self._dim = None
        if initlist:
            self._validate(initlist[0])
        self._list = []
        self.extend(initlist)

    def __len__(self):
        """Length of the list contained in self."""
        return len(self._list)

    def __repr__(self):
        """Return repr(self)."""
        return f'{self.__class__.__name__}({repr(self._list)})'

    def __getitem__(self, idx):
        """Return self[idx]."""
        return self._list[idx]

    def __setitem__(self, idx, value):
        """Set self[idx] = value."""
        if self._index_is_valid(idx):
            self._validate(value)
            self._check_length_update(idx, value)
        self._list[idx] = value

    def __delitem__(self, idx):
        """Perform del self[idx]."""
        if self._index_is_valid(idx):
            self._check_length_update(idx, [])
        del self._list[idx]

    def clear(self):
        """Remove all items from the list."""
        self._check_length_update(slice(0, len(self)), [])
        self._list.clear()

    def insert(self, idx, value):
        """Insert an object before index denoted by idx."""
        if isinstance(idx, Integral):
            self._validate(value, single_item=True)
        self._list.insert(idx, value)

    def _index_is_valid(self, idx, allow_int_only=False):
        """
        Object to be used as list index is within range of
        list contained within self.

        Parameters
        ----------
        idx : object
            List index. Usually an integer type or slice.
        allow_int_only : bool, optional
            Being an integral type is a necessary condition
            for validity. The default is True.

        Returns
        -------
        : bool
            True if index is valid, False otherwise.
        """
        try:
            self._list[idx]
        except (TypeError, IndexError):
            slice_valid = False
        else:
            slice_valid = True
        int_req_satisfied = not allow_int_only or isinstance(idx, Integral)
        return slice_valid and int_req_satisfied

    def _check_length_update(self, idx, value):
        """
        Check whether the update ``self[idx] = value`` reduces the
        length of self to a value smaller than the minimum length.

        Raises
        ------
        ValueError
            If minimum length requirement is violated by the update.
        """
        if isinstance(idx, Integral):
            slice_len = 1
        else:
            slice_len = len(self._list[idx])
        val_len = len(value) if isinstance(value, Iterable) else 1
        new_len = len(self) + val_len - slice_len
        if new_len < self._min_length:
            raise ValueError(f'Length of uncertainty set list {self._name!r} must be at least {self._min_length}')

    def _validate(self, value, single_item=False):
        """
        Validate item or sequence of items to be inserted into self.

        Parameters
        ----------
        value : object
            Object to validate.
        single_item : bool, optional
            Do not allow validation of iterables of objects
            (e.g. a list of ``UncertaintySet`` objects).
            The default is `False`.

        Raises
        ------
        TypeError
            If object passed is not of the appropriate type
            (``UncertaintySet``, or an iterable thereof).
        ValueError
            If object passed is (or contains) an ``UncertaintySet``
            whose dimension does not match that of other uncertainty
            sets in self.
        """
        if not single_item and isinstance(value, Iterable):
            for val in value:
                self._validate(val, single_item=True)
        else:
            validate_arg_type(self._name, value, UncertaintySet, 'An `UncertaintySet` object', is_entry_of_arg=True)
            if self._dim is None:
                self._dim = value.dim
            elif value.dim != self._dim:
                raise ValueError(f'Uncertainty set list with name {self._name!r} contains UncertaintySet objects of dimension {self._dim}, but attempting to add set of dimension {value.dim}')

    @property
    def dim(self):
        """Dimension of all uncertainty sets contained in self."""
        return self._dim