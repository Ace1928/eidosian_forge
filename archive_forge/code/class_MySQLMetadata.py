from collections import namedtuple
from inspect import isclass
import re
import warnings
from peewee import *
from peewee import _StringField
from peewee import _query_val_transform
from peewee import CommaNodeList
from peewee import SCOPE_VALUES
from peewee import make_snake_case
from peewee import text_type
class MySQLMetadata(Metadata):
    if FIELD_TYPE is None:
        column_map = {}
    else:
        column_map = {FIELD_TYPE.BLOB: TextField, FIELD_TYPE.CHAR: CharField, FIELD_TYPE.DATE: DateField, FIELD_TYPE.DATETIME: DateTimeField, FIELD_TYPE.DECIMAL: DecimalField, FIELD_TYPE.DOUBLE: FloatField, FIELD_TYPE.FLOAT: FloatField, FIELD_TYPE.INT24: IntegerField, FIELD_TYPE.LONG_BLOB: TextField, FIELD_TYPE.LONG: IntegerField, FIELD_TYPE.LONGLONG: BigIntegerField, FIELD_TYPE.MEDIUM_BLOB: TextField, FIELD_TYPE.NEWDECIMAL: DecimalField, FIELD_TYPE.SHORT: IntegerField, FIELD_TYPE.STRING: CharField, FIELD_TYPE.TIMESTAMP: DateTimeField, FIELD_TYPE.TIME: TimeField, FIELD_TYPE.TINY_BLOB: TextField, FIELD_TYPE.TINY: IntegerField, FIELD_TYPE.VAR_STRING: CharField}

    def __init__(self, database, **kwargs):
        if 'password' in kwargs:
            kwargs['passwd'] = kwargs.pop('password')
        super(MySQLMetadata, self).__init__(database, **kwargs)

    def get_column_types(self, table, schema=None):
        column_types = {}
        cursor = self.execute('SELECT * FROM `%s` LIMIT 1' % table)
        for column_description in cursor.description:
            name, type_code = column_description[:2]
            column_types[name] = self.column_map.get(type_code, UnknownField)
        return (column_types, {})