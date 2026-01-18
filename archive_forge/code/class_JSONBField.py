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
class JSONBField(JSONField):
    field_type = 'JSONB'
    Path = JSONBPath

    def db_value(self, value):
        if value is not None:
            if not isinstance(value, Node):
                value = fn.jsonb(self._json_dumps(value))
            return value

    def json(self):
        return fn.json(self)

    def extract(self, *paths):
        paths = [Value(p, converter=False) for p in paths]
        return fn.jsonb_extract(self, *paths)

    def remove(self, *paths):
        if not paths:
            return self.Path(self).remove()
        return fn.jsonb_remove(self, *paths)