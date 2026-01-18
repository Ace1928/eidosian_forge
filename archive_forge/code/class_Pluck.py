from __future__ import annotations
import collections
import copy
import datetime as dt
import decimal
import ipaddress
import math
import numbers
import typing
import uuid
import warnings
from collections.abc import Mapping as _Mapping
from enum import Enum as EnumType
from marshmallow import class_registry, types, utils, validate
from marshmallow.base import FieldABC, SchemaABC
from marshmallow.exceptions import (
from marshmallow.utils import (
from marshmallow.utils import (
from marshmallow.validate import And, Length
from marshmallow.warnings import RemovedInMarshmallow4Warning
class Pluck(Nested):
    """Allows you to replace nested data with one of the data's fields.

    Example: ::

        from marshmallow import Schema, fields

        class ArtistSchema(Schema):
            id = fields.Int()
            name = fields.Str()

        class AlbumSchema(Schema):
            artist = fields.Pluck(ArtistSchema, 'id')


        in_data = {'artist': 42}
        loaded = AlbumSchema().load(in_data) # => {'artist': {'id': 42}}
        dumped = AlbumSchema().dump(loaded)  # => {'artist': 42}

    :param Schema nested: The Schema class or class name (string)
        to nest, or ``"self"`` to nest the :class:`Schema` within itself.
    :param str field_name: The key to pluck a value from.
    :param kwargs: The same keyword arguments that :class:`Nested` receives.
    """

    def __init__(self, nested: SchemaABC | type | str | typing.Callable[[], SchemaABC], field_name: str, **kwargs):
        super().__init__(nested, only=(field_name,), **kwargs)
        self.field_name = field_name

    @property
    def _field_data_key(self):
        only_field = self.schema.fields[self.field_name]
        return only_field.data_key or self.field_name

    def _serialize(self, nested_obj, attr, obj, **kwargs):
        ret = super()._serialize(nested_obj, attr, obj, **kwargs)
        if ret is None:
            return None
        if self.many:
            return utils.pluck(ret, key=self._field_data_key)
        return ret[self._field_data_key]

    def _deserialize(self, value, attr, data, partial=None, **kwargs):
        self._test_collection(value)
        if self.many:
            value = [{self._field_data_key: v} for v in value]
        else:
            value = {self._field_data_key: value}
        return self._load(value, data, partial=partial)