import collections.abc
import datetime
from importlib import import_module
import operator
from os import fspath
from os.path import isfile, isdir
import re
import sys
from types import FunctionType, MethodType, ModuleType
import uuid
import warnings
from .constants import DefaultValue, TraitKind, ValidateTrait
from .trait_base import (
from .trait_converters import trait_from, trait_cast
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListEvent, TraitListObject
from .trait_set_object import TraitSetEvent, TraitSetObject
from .trait_type import (
from .traits import (
from .util.deprecated import deprecated
from .util.import_symbol import import_symbol
from .editor_factories import (
class BaseRange(TraitType):
    """ A trait type whose numeric value lies inside a range.

    The value held will be either an integer or a float, which type is
    determined by whether the *low*, *high* and *value* arguments are
    integers or floats.

    The *low*, *high*, and *value* arguments must be of the same type
    (integer or float), except in the case where either *low* or *high* is
    a string (i.e. extended trait name).

    If *value* is None or omitted, the default value is *low*, unless *low*
    is None or omitted, in which case the default value is *high*.

    Parameters
    ----------
    low : integer, float or string (i.e. extended trait name)
        The low end of the range.
    high : integer, float or string (i.e. extended trait name)
        The high end of the range.
    value : integer, float or string (i.e. extended trait name)
        The default value of the trait.
    exclude_low : bool
        Indicates whether the low end of the range is exclusive.
    exclude_high : bool
        Indicates whether the high end of the range is exclusive.
    """
    default_value_type = DefaultValue.constant

    def __init__(self, low=None, high=None, value=None, exclude_low=False, exclude_high=False, **metadata):
        if value is None:
            if low is not None:
                value = low
            else:
                value = high
        super().__init__(value, **metadata)
        vtype = type(high)
        if low is not None and (not issubclass(vtype, (float, str))):
            vtype = type(low)
        is_static = not issubclass(vtype, str)
        if is_static and vtype not in RangeTypes:
            raise TraitError('Range can only be use for int or float values, but a value of type %s was specified.' % vtype)
        self._low_name = self._high_name = ''
        self._vtype = Undefined
        kind = None
        if vtype is float:
            self._validate = 'float_validate'
            kind = ValidateTrait.float_range
            self._type_desc = 'a floating point number'
            if low is not None:
                low = float(low)
            if high is not None:
                high = float(high)
        elif vtype is int:
            self._validate = 'int_validate'
            self._type_desc = 'an integer'
            if low is not None:
                low = int(low)
            if high is not None:
                high = int(high)
        else:
            self.get, self.set, self.validate = (self._get, self._set, self._validate)
            self._vtype = None
            self._type_desc = 'a number'
            if isinstance(high, str):
                self._high_name = high = 'object.' + high
            else:
                self._vtype = type(high)
            high = compile(str(high), '<string>', 'eval')
            if isinstance(low, str):
                self._low_name = low = 'object.' + low
            else:
                self._vtype = type(low)
            low = compile(str(low), '<string>', 'eval')
            if isinstance(value, str):
                value = 'object.' + value
            self._value = compile(str(value), '<string>', 'eval')
            self.default_value_type = DefaultValue.callable
            self.default_value = self._get_default_value
        exclude_mask = 0
        if exclude_low:
            exclude_mask |= 1
        if exclude_high:
            exclude_mask |= 2
        if is_static and kind is not None:
            self.init_fast_validate(kind, low, high, exclude_mask)
        self._low = low
        self._high = high
        self._exclude_low = exclude_low
        self._exclude_high = exclude_high

    def init_fast_validate(self, *args):
        """ Does nothing for the BaseRange class. Used in the Range class to
        set up the fast validator.
        """
        pass

    def validate(self, object, name, value):
        """ Validate that the value is in the specified range.
        """
        return getattr(self, self._validate)(object, name, value)

    def float_validate(self, object, name, value):
        """ Validate that the value is a float value in the specified range.
        """
        original_value = value
        try:
            value = _validate_float(value)
        except TypeError:
            self.error(object, name, original_value)
        if (self._low is None or (self._exclude_low and self._low < value) or (not self._exclude_low and self._low <= value)) and (self._high is None or (self._exclude_high and self._high > value) or (not self._exclude_high and self._high >= value)):
            return value
        self.error(object, name, original_value)

    def int_validate(self, object, name, value):
        """ Validate that the value is an int value in the specified range.
        """
        original_value = value
        try:
            value = _validate_int(value)
        except TypeError:
            self.error(object, name, original_value)
        if (self._low is None or (self._exclude_low and self._low < value) or (not self._exclude_low and self._low <= value)) and (self._high is None or (self._exclude_high and self._high > value) or (not self._exclude_high and self._high >= value)):
            return value
        self.error(object, name, original_value)

    def _get_default_value(self, object):
        """ Returns the default value of the range.
        """
        return eval(self._value)

    def _get(self, object, name, trait):
        """ Returns the current value of a dynamic range trait.
        """
        cname = '_traits_cache_' + name
        value = object.__dict__.get(cname, Undefined)
        if value is Undefined:
            object.__dict__[cname] = value = eval(self._value)
        low = eval(self._low)
        high = eval(self._high)
        if low is not None and value < low:
            value = low
        elif high is not None and value > high:
            value = high
        return self._typed_value(value, low, high)

    def _set(self, object, name, value):
        """ Sets the current value of a dynamic range trait.
        """
        value = self._validate(object, name, value)
        self._set_value(object, name, value)

    def _validate(self, object, name, value):
        """ Validate a value for a dynamic range trait.
        """
        if not isinstance(value, str):
            try:
                low = eval(self._low)
                high = eval(self._high)
                if low is None and high is None:
                    if isinstance(value, RangeTypes):
                        return value
                else:
                    new_value = self._typed_value(value, low, high)
                    if (low is None or (self._exclude_low and low < new_value) or (not self._exclude_low and low <= new_value)) and (high is None or (self._exclude_high and high > new_value) or (not self._exclude_high and high >= new_value)):
                        return new_value
            except:
                pass
        self.error(object, name, value)

    def _typed_value(self, value, low, high):
        """ Returns the specified value with the correct type for the current
            dynamic range.
        """
        vtype = self._vtype
        if vtype is None:
            if low is not None:
                vtype = type(low)
            elif high is not None:
                vtype = type(high)
            else:
                vtype = lambda x: x
        return vtype(value)

    def _set_value(self, object, name, value):
        """ Sets the specified value as the value of the dynamic range.
        """
        cname = '_traits_cache_' + name
        old = object.__dict__.get(cname, Undefined)
        if old is Undefined:
            old = eval(self._value)
        object.__dict__[cname] = value
        if value != old:
            object.trait_property_changed(name, old, value)

    def full_info(self, object, name, value):
        """ Returns a description of the trait.
        """
        if self._vtype is not Undefined:
            low = eval(self._low)
            high = eval(self._high)
            low, high = (self._typed_value(low, low, high), self._typed_value(high, low, high))
        else:
            low = self._low
            high = self._high
        if low is None:
            if high is None:
                return self._type_desc
            return '%s <%s %s' % (self._type_desc, '='[self._exclude_high:], high)
        elif high is None:
            return '%s >%s %s' % (self._type_desc, '='[self._exclude_low:], low)
        return '%s <%s %s <%s %s' % (low, '='[self._exclude_low:], self._type_desc, '='[self._exclude_high:], high)

    def create_editor(self):
        """ Returns the default UI editor for the trait.
        """
        auto_set = self.auto_set
        if auto_set is None:
            auto_set = True
        from traitsui.api import RangeEditor
        return RangeEditor(self, mode=self.mode or 'auto', cols=self.cols or 3, auto_set=auto_set, enter_set=self.enter_set or False, low_label=self.low or '', high_label=self.high or '', low_name=self._low_name, high_name=self._high_name)