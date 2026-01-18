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
class LSMTable(VirtualModel):

    class Meta:
        extension_module = 'lsm1'
        filename = None

    @classmethod
    def clean_options(cls, options):
        filename = cls._meta.filename
        if not filename:
            raise ValueError('LSM1 extension requires that you specify a filename for the LSM database.')
        elif len(filename) >= 2 and filename[0] != '"':
            filename = '"%s"' % filename
        if not cls._meta.primary_key:
            raise ValueError('LSM1 models must specify a primary-key field.')
        key = cls._meta.primary_key
        if isinstance(key, AutoField):
            raise ValueError('LSM1 models must explicitly declare a primary key field.')
        if not isinstance(key, (TextField, BlobField, IntegerField)):
            raise ValueError('LSM1 key must be a TextField, BlobField, or IntegerField.')
        key._hidden = True
        if isinstance(key, IntegerField):
            data_type = 'UINT'
        elif isinstance(key, BlobField):
            data_type = 'BLOB'
        else:
            data_type = 'TEXT'
        cls._meta.prefix_arguments = [filename, '"%s"' % key.name, data_type]
        if len(cls._meta.sorted_fields) == 2:
            cls._meta._value_field = cls._meta.sorted_fields[1]
        else:
            cls._meta._value_field = None
        return options

    @classmethod
    def load_extension(cls, path='lsm.so'):
        cls._meta.database.load_extension(path)

    @staticmethod
    def slice_to_expr(key, idx):
        if idx.start is not None and idx.stop is not None:
            return key.between(idx.start, idx.stop)
        elif idx.start is not None:
            return key >= idx.start
        elif idx.stop is not None:
            return key <= idx.stop

    @staticmethod
    def _apply_lookup_to_query(query, key, lookup):
        if isinstance(lookup, slice):
            expr = LSMTable.slice_to_expr(key, lookup)
            if expr is not None:
                query = query.where(expr)
            return (query, False)
        elif isinstance(lookup, Expression):
            return (query.where(lookup), False)
        else:
            return (query.where(key == lookup), True)

    @classmethod
    def get_by_id(cls, pk):
        query, is_single = cls._apply_lookup_to_query(cls.select().namedtuples(), cls._meta.primary_key, pk)
        if is_single:
            row = query.get()
            return row[1] if cls._meta._value_field is not None else row
        else:
            return query

    @classmethod
    def set_by_id(cls, key, value):
        if cls._meta._value_field is not None:
            data = {cls._meta._value_field: value}
        elif isinstance(value, tuple):
            data = {}
            for field, fval in zip(cls._meta.sorted_fields[1:], value):
                data[field] = fval
        elif isinstance(value, dict):
            data = value
        elif isinstance(value, cls):
            data = value.__dict__
        data[cls._meta.primary_key] = key
        cls.replace(data).execute()

    @classmethod
    def delete_by_id(cls, pk):
        query, is_single = cls._apply_lookup_to_query(cls.delete(), cls._meta.primary_key, pk)
        return query.execute()