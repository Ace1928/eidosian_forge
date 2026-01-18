from __future__ import annotations
import collections
import re
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any
import numpy as np
import scipy.constants as const
class ArrayWithUnit(np.ndarray):
    """Subclasses numpy.ndarray to attach a unit type. Typically, you should
    use the pre-defined unit type subclasses such as EnergyArray,
    LengthArray, etc. instead of using ArrayWithFloatWithUnit directly.

    Supports conversion, addition and subtraction of the same unit type. E.g.,
    1 m + 20 cm will be automatically converted to 1.2 m (units follow the
    leftmost quantity).

    >>> a = EnergyArray([1, 2], "Ha")
    >>> b = EnergyArray([1, 2], "eV")
    >>> c = a + b
    >>> print(c)
    [ 1.03674933  2.07349865] Ha
    >>> c.to("eV")
    array([ 28.21138386,  56.42276772]) eV
    """

    def __new__(cls, input_array, unit, unit_type=None) -> Self:
        """Override __new__."""
        obj = np.asarray(input_array).view(cls)
        obj._unit = Unit(unit)
        obj._unit_type = unit_type
        return obj

    def __array_finalize__(self, obj):
        """See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html for
        comments.
        """
        if obj is None:
            return
        self._unit = getattr(obj, '_unit', None)
        self._unit_type = getattr(obj, '_unit_type', None)

    @property
    def unit_type(self) -> str:
        """The type of unit. Energy, Charge, etc."""
        return self._unit_type

    @property
    def unit(self) -> str:
        """The unit, e.g., "eV"."""
        return self._unit

    def __reduce__(self):
        reduce = list(super().__reduce__())
        reduce[2] = {'np_state': reduce[2], '_unit': self._unit}
        return tuple(reduce)

    def __setstate__(self, state):
        super().__setstate__(state['np_state'])
        self._unit = state['_unit']

    def __repr__(self) -> str:
        return f'{np.array(self)!r} {self.unit}'

    def __str__(self) -> str:
        return f'{np.array(self)} {self.unit}'

    def __add__(self, other):
        if hasattr(other, 'unit_type'):
            if other.unit_type != self.unit_type:
                raise UnitError('Adding different types of units is not allowed')
            if other.unit != self.unit:
                other = other.to(self.unit)
        return type(self)(np.array(self) + np.array(other), unit_type=self.unit_type, unit=self.unit)

    def __sub__(self, other):
        if hasattr(other, 'unit_type'):
            if other.unit_type != self.unit_type:
                raise UnitError('Subtracting different units is not allowed')
            if other.unit != self.unit:
                other = other.to(self.unit)
        return type(self)(np.array(self) - np.array(other), unit_type=self.unit_type, unit=self.unit)

    def __mul__(self, other):
        if not hasattr(other, 'unit_type'):
            return type(self)(np.array(self) * np.array(other), unit_type=self._unit_type, unit=self._unit)
        return type(self)(np.array(self).__mul__(np.array(other)), unit=self.unit * other.unit)

    def __rmul__(self, other):
        if not hasattr(other, 'unit_type'):
            return type(self)(np.array(self) * np.array(other), unit_type=self._unit_type, unit=self._unit)
        return type(self)(np.array(self) * np.array(other), unit=self.unit * other.unit)

    def __truediv__(self, other):
        if not hasattr(other, 'unit_type'):
            return type(self)(np.array(self) / np.array(other), unit_type=self._unit_type, unit=self._unit)
        return type(self)(np.array(self) / np.array(other), unit=self.unit / other.unit)

    def __neg__(self):
        return type(self)(-np.array(self), unit_type=self.unit_type, unit=self.unit)

    def to(self, new_unit):
        """Conversion to a new_unit.

        Args:
            new_unit:
                New unit type.

        Returns:
            A ArrayWithFloatWithUnit object in the new units.

        Example usage:
        >>> e = EnergyArray([1, 1.1], "Ha")
        >>> e.to("eV")
        array([ 27.21138386,  29.93252225]) eV
        """
        return type(self)(np.array(self) * self.unit.get_conversion_factor(new_unit), unit_type=self.unit_type, unit=new_unit)

    @property
    def as_base_units(self):
        """Returns this ArrayWithUnit in base SI units, including derived units.

        Returns:
            An ArrayWithUnit object in base SI units
        """
        return self.to(self.unit.as_base_units[0])

    @property
    def supported_units(self):
        """Supported units for specific unit type."""
        return ALL_UNITS[self.unit_type]

    def conversions(self):
        """Returns a string showing the available conversions.
        Useful tool in interactive mode.
        """
        return '\n'.join((str(self.to(unit)) for unit in self.supported_units))