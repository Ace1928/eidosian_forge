import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
class SchemaBase:
    """Base class for schema wrappers.

    Each derived class should set the _schema class attribute (and optionally
    the _rootschema class attribute) which is used for validation.
    """
    _schema: Optional[Dict[str, Any]] = None
    _rootschema: Optional[Dict[str, Any]] = None
    _class_is_valid_at_instantiation: bool = True

    def __init__(self, *args: Any, **kwds: Any) -> None:
        if self._schema is None:
            raise ValueError('Cannot instantiate object of type {}: _schema class attribute is not defined.'.format(self.__class__))
        if kwds:
            assert len(args) == 0
        else:
            assert len(args) in [0, 1]
        object.__setattr__(self, '_args', args)
        object.__setattr__(self, '_kwds', kwds)
        if DEBUG_MODE and self._class_is_valid_at_instantiation:
            self.to_dict(validate=True)

    def copy(self, deep: Union[bool, Iterable]=True, ignore: Optional[list]=None) -> Self:
        """Return a copy of the object

        Parameters
        ----------
        deep : boolean or list, optional
            If True (default) then return a deep copy of all dict, list, and
            SchemaBase objects within the object structure.
            If False, then only copy the top object.
            If a list or iterable, then only copy the listed attributes.
        ignore : list, optional
            A list of keys for which the contents should not be copied, but
            only stored by reference.
        """

        def _shallow_copy(obj):
            if isinstance(obj, SchemaBase):
                return obj.copy(deep=False)
            elif isinstance(obj, list):
                return obj[:]
            elif isinstance(obj, dict):
                return obj.copy()
            else:
                return obj

        def _deep_copy(obj, ignore: Optional[list]=None):
            if ignore is None:
                ignore = []
            if isinstance(obj, SchemaBase):
                args = tuple((_deep_copy(arg) for arg in obj._args))
                kwds = {k: _deep_copy(v, ignore=ignore) if k not in ignore else v for k, v in obj._kwds.items()}
                with debug_mode(False):
                    return obj.__class__(*args, **kwds)
            elif isinstance(obj, list):
                return [_deep_copy(v, ignore=ignore) for v in obj]
            elif isinstance(obj, dict):
                return {k: _deep_copy(v, ignore=ignore) if k not in ignore else v for k, v in obj.items()}
            else:
                return obj
        try:
            deep = list(deep)
        except TypeError:
            deep_is_list = False
        else:
            deep_is_list = True
        if deep and (not deep_is_list):
            return _deep_copy(self, ignore=ignore)
        with debug_mode(False):
            copy = self.__class__(*self._args, **self._kwds)
        if deep_is_list:
            assert isinstance(deep, list)
            for attr in deep:
                copy[attr] = _shallow_copy(copy._get(attr))
        return copy

    def _get(self, attr, default=Undefined):
        """Get an attribute, returning default if not present."""
        attr = self._kwds.get(attr, Undefined)
        if attr is Undefined:
            attr = default
        return attr

    def __getattr__(self, attr):
        if attr == '_kwds':
            raise AttributeError()
        if attr in self._kwds:
            return self._kwds[attr]
        else:
            try:
                _getattr = super(SchemaBase, self).__getattr__
            except AttributeError:
                _getattr = super(SchemaBase, self).__getattribute__
            return _getattr(attr)

    def __setattr__(self, item, val):
        self._kwds[item] = val

    def __getitem__(self, item):
        return self._kwds[item]

    def __setitem__(self, item, val):
        self._kwds[item] = val

    def __repr__(self):
        if self._kwds:
            args = ('{}: {!r}'.format(key, val) for key, val in sorted(self._kwds.items()) if val is not Undefined)
            args = '\n' + ',\n'.join(args)
            return '{0}({{{1}\n}})'.format(self.__class__.__name__, args.replace('\n', '\n  '))
        else:
            return '{}({!r})'.format(self.__class__.__name__, self._args[0])

    def __eq__(self, other):
        return type(self) is type(other) and self._args == other._args and (self._kwds == other._kwds)

    def to_dict(self, validate: bool=True, *, ignore: Optional[List[str]]=None, context: Optional[Dict[str, Any]]=None) -> dict:
        """Return a dictionary representation of the object

        Parameters
        ----------
        validate : bool, optional
            If True (default), then validate the output dictionary
            against the schema.
        ignore : list[str], optional
            A list of keys to ignore. It is usually not needed
            to specify this argument as a user.
        context : dict[str, Any], optional
            A context dictionary. It is usually not needed
            to specify this argument as a user.

        Notes
        -----
        Technical: The ignore parameter will *not* be passed to child to_dict
        function calls.

        Returns
        -------
        dict
            The dictionary representation of this object

        Raises
        ------
        SchemaValidationError :
            if validate=True and the dict does not conform to the schema
        """
        if context is None:
            context = {}
        if ignore is None:
            ignore = []
        if self._args and (not self._kwds):
            result = _todict(self._args[0], context=context)
        elif not self._args:
            kwds = self._kwds.copy()
            parsed_shorthand = context.pop('parsed_shorthand', {})
            if 'sort' in parsed_shorthand and ('sort' not in kwds or kwds['type'] not in ['ordinal', Undefined]):
                parsed_shorthand.pop('sort')
            kwds.update({k: v for k, v in parsed_shorthand.items() if kwds.get(k, Undefined) is Undefined})
            kwds = {k: v for k, v in kwds.items() if k not in list(ignore) + ['shorthand']}
            if 'mark' in kwds and isinstance(kwds['mark'], str):
                kwds['mark'] = {'type': kwds['mark']}
            result = _todict(kwds, context=context)
        else:
            raise ValueError('{} instance has both a value and properties : cannot serialize to dict'.format(self.__class__))
        if validate:
            try:
                self.validate(result)
            except jsonschema.ValidationError as err:
                raise SchemaValidationError(self, err) from None
        return result

    def to_json(self, validate: bool=True, indent: Optional[Union[int, str]]=2, sort_keys: bool=True, *, ignore: Optional[List[str]]=None, context: Optional[Dict[str, Any]]=None, **kwargs) -> str:
        """Emit the JSON representation for this object as a string.

        Parameters
        ----------
        validate : bool, optional
            If True (default), then validate the output dictionary
            against the schema.
        indent : int, optional
            The number of spaces of indentation to use. The default is 2.
        sort_keys : bool, optional
            If True (default), sort keys in the output.
        ignore : list[str], optional
            A list of keys to ignore. It is usually not needed
            to specify this argument as a user.
        context : dict[str, Any], optional
            A context dictionary. It is usually not needed
            to specify this argument as a user.
        **kwargs
            Additional keyword arguments are passed to ``json.dumps()``

        Notes
        -----
        Technical: The ignore parameter will *not* be passed to child to_dict
        function calls.

        Returns
        -------
        str
            The JSON specification of the chart object.
        """
        if ignore is None:
            ignore = []
        if context is None:
            context = {}
        dct = self.to_dict(validate=validate, ignore=ignore, context=context)
        return json.dumps(dct, indent=indent, sort_keys=sort_keys, **kwargs)

    @classmethod
    def _default_wrapper_classes(cls) -> Generator[Type['SchemaBase'], None, None]:
        """Return the set of classes used within cls.from_dict()"""
        return _subclasses(SchemaBase)

    @classmethod
    def from_dict(cls, dct: dict, validate: bool=True, _wrapper_classes: Optional[Iterable[Type['SchemaBase']]]=None) -> 'SchemaBase':
        """Construct class from a dictionary representation

        Parameters
        ----------
        dct : dictionary
            The dict from which to construct the class
        validate : boolean
            If True (default), then validate the input against the schema.
        _wrapper_classes : iterable (optional)
            The set of SchemaBase classes to use when constructing wrappers
            of the dict inputs. If not specified, the result of
            cls._default_wrapper_classes will be used.

        Returns
        -------
        obj : Schema object
            The wrapped schema

        Raises
        ------
        jsonschema.ValidationError :
            if validate=True and dct does not conform to the schema
        """
        if validate:
            cls.validate(dct)
        if _wrapper_classes is None:
            _wrapper_classes = cls._default_wrapper_classes()
        converter = _FromDict(_wrapper_classes)
        return converter.from_dict(dct, cls)

    @classmethod
    def from_json(cls, json_string: str, validate: bool=True, **kwargs: Any) -> Any:
        """Instantiate the object from a valid JSON string

        Parameters
        ----------
        json_string : string
            The string containing a valid JSON chart specification.
        validate : boolean
            If True (default), then validate the input against the schema.
        **kwargs :
            Additional keyword arguments are passed to json.loads

        Returns
        -------
        chart : Chart object
            The altair Chart object built from the specification.
        """
        dct = json.loads(json_string, **kwargs)
        return cls.from_dict(dct, validate=validate)

    @classmethod
    def validate(cls, instance: Dict[str, Any], schema: Optional[Dict[str, Any]]=None) -> None:
        """
        Validate the instance against the class schema in the context of the
        rootschema.
        """
        if schema is None:
            schema = cls._schema
        assert schema is not None
        return validate_jsonschema(instance, schema, rootschema=cls._rootschema or cls._schema)

    @classmethod
    def resolve_references(cls, schema: Optional[dict]=None) -> dict:
        """Resolve references in the context of this object's schema or root schema."""
        schema_to_pass = schema or cls._schema
        assert schema_to_pass is not None
        return _resolve_references(schema=schema_to_pass, rootschema=cls._rootschema or cls._schema or schema)

    @classmethod
    def validate_property(cls, name: str, value: Any, schema: Optional[dict]=None) -> None:
        """
        Validate a property against property schema in the context of the
        rootschema
        """
        value = _todict(value, context={})
        props = cls.resolve_references(schema or cls._schema).get('properties', {})
        return validate_jsonschema(value, props.get(name, {}), rootschema=cls._rootschema or cls._schema)

    def __dir__(self) -> list:
        return sorted(list(super().__dir__()) + list(self._kwds.keys()))