import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class _JsonLookupBase(_LookupNode):

    def __init__(self, node, parts, as_json=False):
        super(_JsonLookupBase, self).__init__(node, parts)
        self._as_json = as_json

    def clone(self):
        return type(self)(self.node, list(self.parts), self._as_json)

    @Node.copy
    def as_json(self, as_json=True):
        self._as_json = as_json

    def concat(self, rhs):
        if not isinstance(rhs, Node):
            rhs = Json(rhs)
        return Expression(self.as_json(True), OP.CONCAT, rhs)

    def contains(self, other):
        clone = self.as_json(True)
        if isinstance(other, (list, dict)):
            return Expression(clone, JSONB_CONTAINS, Json(other))
        return Expression(clone, JSONB_EXISTS, other)

    def contains_any(self, *keys):
        return Expression(self.as_json(True), JSONB_CONTAINS_ANY_KEY, Value(list(keys), unpack=False))

    def contains_all(self, *keys):
        return Expression(self.as_json(True), JSONB_CONTAINS_ALL_KEYS, Value(list(keys), unpack=False))

    def has_key(self, key):
        return Expression(self.as_json(True), JSONB_CONTAINS_KEY, key)