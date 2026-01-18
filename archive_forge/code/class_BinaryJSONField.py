import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class BinaryJSONField(IndexedFieldMixin, JSONField):
    field_type = 'JSONB'
    _json_datatype = 'jsonb'
    __hash__ = Field.__hash__

    def contains(self, other):
        if isinstance(other, (list, dict)):
            return Expression(self, JSONB_CONTAINS, Json(other))
        elif isinstance(other, JSONField):
            return Expression(self, JSONB_CONTAINS, other)
        return Expression(cast_jsonb(self), JSONB_EXISTS, other)

    def contained_by(self, other):
        return Expression(cast_jsonb(self), JSONB_CONTAINED_BY, Json(other))

    def contains_any(self, *items):
        return Expression(cast_jsonb(self), JSONB_CONTAINS_ANY_KEY, Value(list(items), unpack=False))

    def contains_all(self, *items):
        return Expression(cast_jsonb(self), JSONB_CONTAINS_ALL_KEYS, Value(list(items), unpack=False))

    def has_key(self, key):
        return Expression(cast_jsonb(self), JSONB_CONTAINS_KEY, key)

    def remove(self, *items):
        return Expression(cast_jsonb(self), JSONB_REMOVE, Value(list(items), unpack=False))