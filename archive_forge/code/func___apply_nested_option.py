from __future__ import annotations
import copy
import datetime as dt
import decimal
import inspect
import json
import typing
import uuid
import warnings
from abc import ABCMeta
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from marshmallow import base, class_registry, types
from marshmallow import fields as ma_fields
from marshmallow.decorators import (
from marshmallow.error_store import ErrorStore
from marshmallow.exceptions import StringNotCollectionError, ValidationError
from marshmallow.orderedset import OrderedSet
from marshmallow.utils import (
from marshmallow.warnings import RemovedInMarshmallow4Warning
def __apply_nested_option(self, option_name, field_names, set_operation) -> None:
    """Apply nested options to nested fields"""
    nested_fields = [name.split('.', 1) for name in field_names if '.' in name]
    nested_options = defaultdict(list)
    for parent, nested_names in nested_fields:
        nested_options[parent].append(nested_names)
    for key, options in iter(nested_options.items()):
        new_options = self.set_class(options)
        original_options = getattr(self.declared_fields[key], option_name, ())
        if original_options:
            if set_operation == 'union':
                new_options |= self.set_class(original_options)
            if set_operation == 'intersection':
                new_options &= self.set_class(original_options)
        setattr(self.declared_fields[key], option_name, new_options)