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
def _bind_field(self, field_name: str, field_obj: ma_fields.Field) -> None:
    """Bind field to the schema, setting any necessary attributes on the
        field (e.g. parent and name).

        Also set field load_only and dump_only values if field_name was
        specified in ``class Meta``.
        """
    if field_name in self.load_only:
        field_obj.load_only = True
    if field_name in self.dump_only:
        field_obj.dump_only = True
    try:
        field_obj._bind_to_schema(field_name, self)
    except TypeError as error:
        if isinstance(field_obj, type) and issubclass(field_obj, base.FieldABC):
            msg = f'Field for "{field_name}" must be declared as a Field instance, not a class. Did you mean "fields.{field_obj.__name__}()"?'
            raise TypeError(msg) from error
        raise error
    self.on_bind_field(field_name, field_obj)