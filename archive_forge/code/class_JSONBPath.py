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
class JSONBPath(JSONPath):

    def append(self, value, as_json=None):
        if as_json or isinstance(value, (list, dict)):
            value = fn.jsonb(self._field._json_dumps(value))
        return fn.jsonb_set(self._field, self['#'].path, value)

    def _json_operation(self, func, value, as_json=None):
        if as_json or isinstance(value, (list, dict)):
            value = fn.jsonb(self._field._json_dumps(value))
        return func(self._field, self.path, value)

    def insert(self, value, as_json=None):
        return self._json_operation(fn.jsonb_insert, value, as_json)

    def set(self, value, as_json=None):
        return self._json_operation(fn.jsonb_set, value, as_json)

    def replace(self, value, as_json=None):
        return self._json_operation(fn.jsonb_replace, value, as_json)

    def update(self, value):
        return self.set(fn.jsonb_patch(self, self._field._json_dumps(value)))

    def remove(self):
        return fn.jsonb_remove(self._field, self.path)

    def __sql__(self, ctx):
        return ctx.sql(fn.jsonb_extract(self._field, self.path) if self._path else self._field)