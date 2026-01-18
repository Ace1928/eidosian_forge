import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
class PyomoUnitsContainer(object):
    """Class that is used to create and contain units in Pyomo.

    This is the class that is used to create, contain, and interact
    with units in Pyomo.  The module
    (:mod:`pyomo.core.base.units_container`) also contains a module
    level units container :py:data:`units` that is an instance of a
    PyomoUnitsContainer. This module instance should typically be used
    instead of creating your own instance of a
    :py:class:`PyomoUnitsContainer`.  For an overview of the usage of
    this class, see the module documentation
    (:mod:`pyomo.core.base.units_container`)

    This class is based on the "pint" module. Documentation for
    available units can be found at the following url:
    https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

    .. note::

        Pre-defined units can be accessed through attributes on the
        PyomoUnitsContainer class; however, these attributes are created
        dynamically through the __getattr__ method, and are not present
        on the class until they are requested.

    """

    def __init__(self, pint_registry=NOTSET):
        """Create a PyomoUnitsContainer instance."""
        if pint_registry is NOTSET:
            pint_registry = pint_module.UnitRegistry()
        self._pint_registry = pint_registry
        if pint_registry is None:
            self._pint_dimensionless = None
        else:
            self._pint_dimensionless = self._pint_registry.dimensionless
        self._pintUnitExtractionVisitor = PintUnitExtractionVisitor(self)

    def load_definitions_from_file(self, definition_file):
        """Load new units definitions from a file

        This method loads additional units definitions from a user
        specified definition file. An example of a definitions file
        can be found at:
        https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

        If we have a file called ``my_additional_units.txt`` with the
        following lines::

            USD = [currency]

        Then we can add this to the container with:

        .. doctest::
            :skipif: not pint_available
            :hide:

            # Get a local units object (to avoid duplicate registration
            # with the example in load_definitions_from_strings)
            >>> import pyomo.core.base.units_container as _units
            >>> u = _units.PyomoUnitsContainer()
            >>> with open('my_additional_units.txt', 'w') as FILE:
            ...     tmp = FILE.write("USD = [currency]\\n")

        .. doctest::
            :skipif: not pint_available

            >>> u.load_definitions_from_file('my_additional_units.txt')
            >>> print(u.USD)
            USD

        .. doctest::
            :skipif: not pint_available
            :hide:

            # Clean up the file we just created
            >>> import os
            >>> os.remove('my_additional_units.txt')

        """
        self._pint_registry.load_definitions(definition_file)
        self._pint_dimensionless = self._pint_registry.dimensionless

    def load_definitions_from_strings(self, definition_string_list):
        """Load new units definitions from a string

        This method loads additional units definitions from a list of
        strings (one for each line). An example of the definitions
        strings can be found at:
        https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

        For example, to add the currency dimension and US dollars as a
        unit, use

        .. doctest::
            :skipif: not pint_available
            :hide:

            # get a local units object (to avoid duplicate registration
            # with the example in load_definitions_from_strings)
            >>> import pint
            >>> import pyomo.core.base.units_container as _units
            >>> u = _units.PyomoUnitsContainer()

        .. doctest::
            :skipif: not pint_available

            >>> u.load_definitions_from_strings(['USD = [currency]'])
            >>> print(u.USD)
            USD

        """
        self._pint_registry.load_definitions(definition_string_list)

    def __getattr__(self, item):
        """Here, __getattr__ is implemented to automatically create the
        necessary unit if the attribute does not already exist.

        Parameters
        ----------
        item : str
            the name of the new field requested external

        Returns
        -------
        PyomoUnit
            returns a PyomoUnit corresponding to the requested attribute,
            or None if it cannot be created.

        """
        pint_registry = self._pint_registry
        try:
            pint_unit = getattr(pint_registry, item)
            if pint_unit is not None:
                pint_unit_container = pint_module.util.to_units_container(pint_unit, pint_registry)
                for u, e in pint_unit_container.items():
                    if not pint_registry._units[u].is_multiplicative:
                        raise UnitsError(f'Pyomo units system does not support the offset units "{item}". Use absolute units (e.g. kelvin instead of degC) instead.')
                unit = _PyomoUnit(pint_unit, pint_registry)
                setattr(self, item, unit)
                return unit
        except pint_module.errors.UndefinedUnitError as exc:
            pint_unit = None
        if pint_unit is None:
            raise AttributeError(f'Attribute {item} not found.')

    def _rel_diff(self, a, b):
        scale = min(abs(a), abs(b))
        if scale < 1.0:
            scale = 1.0
        return abs(a - b) / scale

    def _equivalent_pint_units(self, a, b, TOL=1e-12):
        if a is b or a == b:
            return True
        base_a = self._pint_registry.get_base_units(a)
        base_b = self._pint_registry.get_base_units(b)
        if base_a[1] != base_b[1]:
            uc_a = base_a[1].dimensionality
            uc_b = base_b[1].dimensionality
            for key in uc_a.keys() | uc_b.keys():
                if self._rel_diff(uc_a.get(key, 0), uc_b.get(key, 0)) >= TOL:
                    return False
        return self._rel_diff(base_a[0], base_b[0]) <= TOL

    def _equivalent_to_dimensionless(self, a, TOL=1e-12):
        if a is self._pint_dimensionless or a == self._pint_dimensionless:
            return True
        base_a = self._pint_registry.get_base_units(a)
        if not base_a[1].dimensionless:
            return False
        return self._rel_diff(base_a[0], 1.0) <= TOL

    def _get_pint_units(self, expr):
        """
        Return the pint units corresponding to the expression. This does
        a number of checks as well.

        Parameters
        ----------
        expr : Pyomo expression
           the input expression for extracting units

        Returns
        -------
        : pint unit
        """
        if expr is None:
            return self._pint_dimensionless
        return self._pintUnitExtractionVisitor.walk_expression(expr=expr)

    def get_units(self, expr):
        """Return the Pyomo units corresponding to this expression (also
        performs validation and will raise an exception if units are not
        consistent).

        Parameters
        ----------
        expr : Pyomo expression
            The expression containing the desired units

        Returns
        -------
        : Pyomo unit (expression)
           Returns the units corresponding to the expression

        Raises
        ------
        :py:class:`pyomo.core.base.units_container.UnitsError`, :py:class:`pyomo.core.base.units_container.InconsistentUnitsError`

        """
        return _PyomoUnit(self._get_pint_units(expr), self._pint_registry)

    def _pint_convert_temp_from_to(self, numerical_value, pint_from_units, pint_to_units):
        if type(numerical_value) not in native_numeric_types:
            raise UnitsError('Conversion routines for absolute and relative temperatures require a numerical value only. Pyomo objects (Var, Param, expressions) are not supported. Please use value(x) to extract the numerical value if necessary.')
        src_quantity = self._pint_registry.Quantity(numerical_value, pint_from_units)
        dest_quantity = src_quantity.to(pint_to_units)
        return dest_quantity.magnitude

    def convert_temp_K_to_C(self, value_in_K):
        """
        Convert a value in Kelvin to degrees Celsius.  Note that this method
        converts a numerical value only. If you need temperature
        conversions in expressions, please work in absolute
        temperatures only.
        """
        return self._pint_convert_temp_from_to(value_in_K, self._pint_registry.K, self._pint_registry.degC)

    def convert_temp_C_to_K(self, value_in_C):
        """
        Convert a value in degrees Celsius to Kelvin Note that this
        method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
        return self._pint_convert_temp_from_to(value_in_C, self._pint_registry.degC, self._pint_registry.K)

    def convert_temp_R_to_F(self, value_in_R):
        """
        Convert a value in Rankine to degrees Fahrenheit.  Note that
        this method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
        return self._pint_convert_temp_from_to(value_in_R, self._pint_registry.rankine, self._pint_registry.degF)

    def convert_temp_F_to_R(self, value_in_F):
        """
        Convert a value in degrees Fahrenheit to Rankine.  Note that
        this method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
        return self._pint_convert_temp_from_to(value_in_F, self._pint_registry.degF, self._pint_registry.rankine)

    def convert(self, src, to_units=None):
        """
        This method returns an expression that contains the
        explicit conversion from one unit to another.

        Parameters
        ----------
        src : Pyomo expression
           The source value that will be converted. This could be a
           Pyomo Var, Pyomo Param, or a more complex expression.
        to_units : Pyomo units expression
           The desired target units for the new expression

        Returns
        -------
           ret : Pyomo expression
        """
        src_pint_unit = self._get_pint_units(src)
        to_pint_unit = self._get_pint_units(to_units)
        if src_pint_unit == to_pint_unit:
            return src
        src_base_factor, base_units_src = self._pint_registry.get_base_units(src_pint_unit, check_nonmult=True)
        to_base_factor, base_units_to = self._pint_registry.get_base_units(to_pint_unit, check_nonmult=True)
        if base_units_src != base_units_to:
            raise InconsistentUnitsError(src_pint_unit, to_pint_unit, 'Error in convert: units not compatible.')
        return src_base_factor / to_base_factor * _PyomoUnit(to_pint_unit / src_pint_unit, self._pint_registry) * src

    def convert_value(self, num_value, from_units=None, to_units=None):
        """
        This method performs explicit conversion of a numerical value
        from one unit to another, and returns the new value.

        The argument "num_value" must be a native numeric type (e.g. float).
        Note that this method returns a numerical value only, and not an
        expression with units.

        Parameters
        ----------
        num_value : float or other native numeric type
           The value that will be converted
        from_units : Pyomo units expression
           The units to convert from
        to_units : Pyomo units expression
           The units to convert to

        Returns
        -------
           float : The converted value

        """
        if type(num_value) not in native_numeric_types:
            raise UnitsError('The argument "num_value" in convert_value must be a native numeric type, but instead type {type(num_value)} was found.')
        from_pint_unit = self._get_pint_units(from_units)
        to_pint_unit = self._get_pint_units(to_units)
        if from_pint_unit == to_pint_unit:
            return num_value
        from_base_factor, from_base_units = self._pint_registry.get_base_units(from_pint_unit, check_nonmult=True)
        to_base_factor, to_base_units = self._pint_registry.get_base_units(to_pint_unit, check_nonmult=True)
        if from_base_units != to_base_units:
            raise UnitsError('Cannot convert %s to %s. Units are not compatible.' % (from_units, to_units))
        from_quantity = num_value * from_pint_unit
        to_quantity = from_quantity.to(to_pint_unit)
        return to_quantity.magnitude

    def set_pint_registry(self, pint_registry):
        if pint_registry is self._pint_registry:
            return
        if self._pint_registry is not None:
            logger.warning('Changing the pint registry used by the Pyomo Units system after the PyomoUnitsContainer was constructed.  Pint requires that all units and dimensioned quantities are generated by a single pint registry.')
        self._pint_registry = pint_registry
        self._pint_dimensionless = self._pint_registry.dimensionless

    @property
    def pint_registry(self):
        return self._pint_registry