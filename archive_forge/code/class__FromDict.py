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
class _FromDict:
    """Class used to construct SchemaBase class hierarchies from a dict

    The primary purpose of using this class is to be able to build a hash table
    that maps schemas to their wrapper classes. The candidate classes are
    specified in the ``class_list`` argument to the constructor.
    """
    _hash_exclude_keys = ('definitions', 'title', 'description', '$schema', 'id')

    def __init__(self, class_list: Iterable[Type[SchemaBase]]) -> None:
        self.class_dict = collections.defaultdict(list)
        for cls in class_list:
            if cls._schema is not None:
                self.class_dict[self.hash_schema(cls._schema)].append(cls)

    @classmethod
    def hash_schema(cls, schema: dict, use_json: bool=True) -> int:
        """
        Compute a python hash for a nested dictionary which
        properly handles dicts, lists, sets, and tuples.

        At the top level, the function excludes from the hashed schema all keys
        listed in `exclude_keys`.

        This implements two methods: one based on conversion to JSON, and one based
        on recursive conversions of unhashable to hashable types; the former seems
        to be slightly faster in several benchmarks.
        """
        if cls._hash_exclude_keys and isinstance(schema, dict):
            schema = {key: val for key, val in schema.items() if key not in cls._hash_exclude_keys}
        if use_json:
            s = json.dumps(schema, sort_keys=True)
            return hash(s)
        else:

            def _freeze(val):
                if isinstance(val, dict):
                    return frozenset(((k, _freeze(v)) for k, v in val.items()))
                elif isinstance(val, set):
                    return frozenset(map(_freeze, val))
                elif isinstance(val, list) or isinstance(val, tuple):
                    return tuple(map(_freeze, val))
                else:
                    return val
            return hash(_freeze(schema))

    def from_dict(self, dct: dict, cls: Optional[Type[SchemaBase]]=None, schema: Optional[dict]=None, rootschema: Optional[dict]=None, default_class=_passthrough) -> Any:
        """Construct an object from a dict representation"""
        if (schema is None) == (cls is None):
            raise ValueError('Must provide either cls or schema, but not both.')
        if schema is None:
            schema = cls._schema
            assert schema is not None
            if rootschema:
                rootschema = rootschema
            elif cls is not None and cls._rootschema is not None:
                rootschema = cls._rootschema
            else:
                rootschema = None
        rootschema = rootschema or schema
        if isinstance(dct, SchemaBase):
            return dct
        if cls is None:
            matches = self.class_dict[self.hash_schema(schema)]
            if matches:
                cls = matches[0]
            else:
                cls = default_class
        schema = _resolve_references(schema, rootschema)
        if 'anyOf' in schema or 'oneOf' in schema:
            schemas = schema.get('anyOf', []) + schema.get('oneOf', [])
            for possible_schema in schemas:
                try:
                    validate_jsonschema(dct, possible_schema, rootschema=rootschema)
                except jsonschema.ValidationError:
                    continue
                else:
                    return self.from_dict(dct, schema=possible_schema, rootschema=rootschema, default_class=cls)
        if isinstance(dct, dict):
            props = schema.get('properties', {})
            kwds = {}
            for key, val in dct.items():
                if key in props:
                    val = self.from_dict(val, schema=props[key], rootschema=rootschema)
                kwds[key] = val
            return cls(**kwds)
        elif isinstance(dct, list):
            item_schema = schema.get('items', {})
            dct = [self.from_dict(val, schema=item_schema, rootschema=rootschema) for val in dct]
            return cls(dct)
        else:
            return cls(dct)