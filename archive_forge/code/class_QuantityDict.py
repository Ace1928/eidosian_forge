import copy
from .chemistry import Substance
from .units import (
from .util.arithmeticdict import ArithmeticDict, _imul, _itruediv
from .printing import as_per_substance_html_table
class QuantityDict(ArithmeticDict):

    def __init__(self, units, *args, **kwargs):
        self.units = units
        super(QuantityDict, self).__init__(lambda: 0 * self.units, *args, **kwargs)
        self._check()

    @classmethod
    def of_quantity(cls, quantity_name, *args, **kwargs):
        instance = cls(get_derived_unit(SI_base_registry, quantity_name), *args, **kwargs)
        instance.quantity_name = quantity_name
        return instance

    def rescale(self, new_units):
        return self.__class__(new_units, {k: rescale(v, new_units) for k, v in self.items()})

    def _repr_html_(self):
        if hasattr(self, 'quantity_name'):
            header = self.quantity_name.capitalize() + ' / '
        else:
            header = ''
        header += html_of_unit(self.units)
        tab = as_per_substance_html_table(to_unitless(self, self.units), header=header)
        return tab._repr_html_()

    def _check(self):
        for k, v in self.items():
            if not is_unitless(v / self.units):
                raise ValueError('entry for %s (%s) is not compatible with %s' % (k, v, self.units))

    def __setitem__(self, key, value):
        if not is_unitless(value / self.units):
            raise ValueError('entry for %s (%s) is not compatible with %s' % (key, value, self.units))
        super(QuantityDict, self).__setitem__(key, value)

    def copy(self):
        return self.__class__(self.units, copy.deepcopy(list(self.items())))

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, repr(self.units), dict(self))

    def __mul__(self, other):
        d = dict(copy.deepcopy(list(self.items())))
        _imul(d, other)
        return self.__class__(self.units * getattr(other, 'units', 1), d)

    def __truediv__(self, other):
        d = dict(copy.deepcopy(list(self.items())))
        _itruediv(d, other)
        return self.__class__(self.units / getattr(other, 'units', 1), d)

    def __floordiv__(self, other):
        a = self.copy()
        if getattr(other, 'units', 1) != 1:
            raise ValueError('Floor division with quantities not defined')
        a //= other
        return a

    def __rtruediv__(self, other):
        """other / self"""
        return self.__class__(getattr(other, 'units', 1) / self.units, {k: other / v for k, v in self.items()})

    def __rfloordiv__(self, other):
        """other // self"""
        return self.__class__(getattr(other, 'units', 1) / self.units, {k: other // v for k, v in self.items()})