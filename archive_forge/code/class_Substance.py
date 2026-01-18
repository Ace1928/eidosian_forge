from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import chain, product
from operator import mul, add
import copy
import math
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util._expr import Expr
from .util.periodic import mass_from_composition
from .util.parsing import (
from .units import default_units, is_quantity, unit_of, to_unitless
from ._util import intdiv
from .util.pyutil import deprecated, DeferredImport, ChemPyDeprecationWarning
class Substance(object):
    """ Class representing a chemical substance

    Parameters
    ----------
    name : str
    charge : int (optional, default: None)
        Will be stored in composition[0], prefer composition when possible.
    latex_name : str
    unicode_name : str
    html_name : str
    composition : dict or None (default)
        Dictionary (int -> number) e.g. {atomic number: count}, zero has special
        meaning (net charge). Avoid using the key 0 unless you specifically mean
        net charge. The motivation behind this is that it is easier to track a
        net-charge of e.g. 6 for U(VI) than it is to remember that uranium has 92
        electrons and use 86 as the value).
    data : dict
        Free form dictionary. Could be simple such as ``{'mp': 0, 'bp': 100}``
        or considerably more involved, e.g.: ``{'diffusion_coefficient': { 'water': lambda T: 2.1*m**2/s/K*(T - 273.15*K)}}``.

    Attributes
    ----------
    mass
        Maps to data['mass'], and when unavailable looks for ``formula.mass``.
    attrs
        A tuple of attribute names for serialization.
    composition : dict or None
        Dictionary mapping fragment key (str) to amount (int).
    data
        Free form dictionary.

    Examples
    --------
    >>> ammonium = Substance('NH4+', 1, 'NH_4^+', composition={7: 1, 1: 4},
    ...     data={'mass': 18.0385, 'pKa': 9.24})
    >>> ammonium.name
    'NH4+'
    >>> ammonium.composition == {0: 1, 1: 4, 7: 1}  # charge represented by key '0'
    True
    >>> ammonium.data['mass']
    18.0385
    >>> ammonium.data['pKa']
    9.24
    >>> ammonium.mass  # mass is a special case (also attribute)
    18.0385
    >>> ammonium.pKa
    Traceback (most recent call last):
        ...
    AttributeError: 'Substance' object has no attribute 'pKa'
    >>> nh4p = Substance.from_formula('NH4+')  # simpler
    >>> nh4p.composition == {7: 1, 1: 4, 0: 1}
    True
    >>> nh4p.latex_name
    'NH_{4}^{+}'

    """
    attrs = ('name', 'latex_name', 'unicode_name', 'html_name', 'composition', 'data')

    def __eq__(self, other):
        for attr in self.attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    @property
    def charge(self):
        """Convenience property for accessing ``composition[0]``"""
        return self.composition.get(0, 0)

    @property
    def mass(self):
        """Convenience property for accessing ``data['mass']``

        when ``data['mass']`` is missing the mass is calculated
        from the :attr:`composition` using
        :func:`chempy.util.parsing.mass_from_composition`.
        """
        try:
            return self.data['mass']
        except KeyError:
            if self.composition is not None:
                return mass_from_composition(self.composition)

    @mass.setter
    def mass(self, value):
        self.data['mass'] = value

    def molar_mass(self, units=None):
        """Returns the molar mass (with units) of the substance

        Examples
        --------
        >>> nh4p = Substance.from_formula('NH4+')  # simpler
        >>> from chempy.units import default_units as u
        >>> nh4p.molar_mass(u)
        array(18.0384511...) * g/mol

        """
        if units is None:
            units = default_units
        return self.mass * units.g / units.mol

    def __init__(self, name=None, charge=None, latex_name=None, unicode_name=None, html_name=None, composition=None, data=None):
        self.name = name
        self.latex_name = latex_name
        self.unicode_name = unicode_name
        self.html_name = html_name
        self.composition = composition
        if self.composition is not None and 0 in self.composition:
            if charge is not None:
                raise KeyError('Cannot give both charge and composition[0]')
        elif charge is not None and composition is not None:
            self.composition[0] = charge
        self.data = data or {}

    @classmethod
    def from_formula(cls, formula, **kwargs):
        """Creates a :class:`Substance` instance from its formula

        Parameters
        ----------
        formula: str
            e.g. 'Na+', 'H2O', 'Fe(CN)6-4'
        \\*\\*kwargs:
            keyword arguments passed on to `.Substance`

        Examples
        --------
        >>> NH3 = Substance.from_formula('NH3')
        >>> NH3.composition == {1: 3, 7: 1}
        True
        >>> '%.2f' % NH3.mass
        '17.03'
        >>> NH3.charge
        0
        >>> NH3.latex_name
        'NH_{3}'

        """
        return cls(formula, latex_name=formula_to_latex(formula), unicode_name=formula_to_unicode(formula), html_name=formula_to_html(formula), composition=formula_to_composition(formula), **kwargs)

    def __repr__(self):
        kw = ['name=' + self.name + ', ...']
        return '<{}({})>'.format(self.__class__.__name__, ','.join(kw))

    def __str__(self):
        return str(self.name)

    def _repr_html_(self):
        return self.html_name

    @staticmethod
    def composition_keys(substance_iter, skip_keys=()):
        """Occurring :attr:`composition` keys among a series of substances"""
        keys = set()
        for s in substance_iter:
            if s.composition is None:
                continue
            for k in s.composition.keys():
                if k in skip_keys:
                    continue
                keys.add(k)
        return sorted(keys)