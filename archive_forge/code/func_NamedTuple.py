import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
@_ensure_subclassable(_namedtuple_mro_entries)
def NamedTuple(typename, fields=_marker, /, **kwargs):
    """Typed version of namedtuple.

        Usage::

            class Employee(NamedTuple):
                name: str
                id: int

        This is equivalent to::

            Employee = collections.namedtuple('Employee', ['name', 'id'])

        The resulting class has an extra __annotations__ attribute, giving a
        dict that maps field names to types.  (The field names are also in
        the _fields attribute, which is part of the namedtuple API.)
        An alternative equivalent functional syntax is also accepted::

            Employee = NamedTuple('Employee', [('name', str), ('id', int)])
        """
    if fields is _marker:
        if kwargs:
            deprecated_thing = 'Creating NamedTuple classes using keyword arguments'
            deprecation_msg = '{name} is deprecated and will be disallowed in Python {remove}. Use the class-based or functional syntax instead.'
        else:
            deprecated_thing = "Failing to pass a value for the 'fields' parameter"
            example = f'`{typename} = NamedTuple({typename!r}, [])`'
            deprecation_msg = '{name} is deprecated and will be disallowed in Python {remove}. To create a NamedTuple class with 0 fields using the functional syntax, pass an empty list, e.g. ' + example + '.'
    elif fields is None:
        if kwargs:
            raise TypeError("Cannot pass `None` as the 'fields' parameter and also specify fields using keyword arguments")
        else:
            deprecated_thing = "Passing `None` as the 'fields' parameter"
            example = f'`{typename} = NamedTuple({typename!r}, [])`'
            deprecation_msg = '{name} is deprecated and will be disallowed in Python {remove}. To create a NamedTuple class with 0 fields using the functional syntax, pass an empty list, e.g. ' + example + '.'
    elif kwargs:
        raise TypeError('Either list of fields or keywords can be provided to NamedTuple, not both')
    if fields is _marker or fields is None:
        warnings.warn(deprecation_msg.format(name=deprecated_thing, remove='3.15'), DeprecationWarning, stacklevel=2)
        fields = kwargs.items()
    nt = _make_nmtuple(typename, fields, module=_caller())
    nt.__orig_bases__ = (NamedTuple,)
    return nt