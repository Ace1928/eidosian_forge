import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
class JSONPath(ColumnBase):

    def __init__(self, field, path=None):
        super(JSONPath, self).__init__()
        self._field = field
        self._path = path or ()

    @property
    def path(self):
        return Value('$%s' % ''.join(self._path))

    def __getitem__(self, idx):
        if isinstance(idx, int) or idx == '#':
            item = '[%s]' % idx
        else:
            item = '.%s' % idx
        return type(self)(self._field, self._path + (item,))

    def append(self, value, as_json=None):
        if as_json or isinstance(value, (list, dict)):
            value = fn.json(self._field._json_dumps(value))
        return fn.json_set(self._field, self['#'].path, value)

    def _json_operation(self, func, value, as_json=None):
        if as_json or isinstance(value, (list, dict)):
            value = fn.json(self._field._json_dumps(value))
        return func(self._field, self.path, value)

    def insert(self, value, as_json=None):
        return self._json_operation(fn.json_insert, value, as_json)

    def set(self, value, as_json=None):
        return self._json_operation(fn.json_set, value, as_json)

    def replace(self, value, as_json=None):
        return self._json_operation(fn.json_replace, value, as_json)

    def update(self, value):
        return self.set(fn.json_patch(self, self._field._json_dumps(value)))

    def remove(self):
        return fn.json_remove(self._field, self.path)

    def json_type(self):
        return fn.json_type(self._field, self.path)

    def length(self):
        return fn.json_array_length(self._field, self.path)

    def children(self):
        return fn.json_each(self._field, self.path)

    def tree(self):
        return fn.json_tree(self._field, self.path)

    def __sql__(self, ctx):
        return ctx.sql(fn.json_extract(self._field, self.path) if self._path else self._field)