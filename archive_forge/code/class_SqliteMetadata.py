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
class SqliteMetadata(Metadata):
    column_map = {'bigint': BigIntegerField, 'blob': BlobField, 'bool': BooleanField, 'boolean': BooleanField, 'char': CharField, 'date': DateField, 'datetime': DateTimeField, 'decimal': DecimalField, 'float': FloatField, 'integer': IntegerField, 'integer unsigned': IntegerField, 'int': IntegerField, 'long': BigIntegerField, 'numeric': DecimalField, 'real': FloatField, 'smallinteger': IntegerField, 'smallint': IntegerField, 'smallint unsigned': IntegerField, 'text': TextField, 'time': TimeField, 'varchar': CharField}
    begin = '(?:["\\[\\(]+)?'
    end = '(?:["\\]\\)]+)?'
    re_foreign_key = '(?:FOREIGN KEY\\s*)?{begin}(.+?){end}\\s+(?:.+\\s+)?references\\s+{begin}(.+?){end}\\s*\\(["|\\[]?(.+?)["|\\]]?\\)'.format(begin=begin, end=end)
    re_varchar = '^\\s*(?:var)?char\\s*\\(\\s*(\\d+)\\s*\\)\\s*$'

    def _map_col(self, column_type):
        raw_column_type = column_type.lower()
        if raw_column_type in self.column_map:
            field_class = self.column_map[raw_column_type]
        elif re.search(self.re_varchar, raw_column_type):
            field_class = CharField
        else:
            column_type = re.sub('\\(.+\\)', '', raw_column_type)
            if column_type == '':
                field_class = BareField
            else:
                field_class = self.column_map.get(column_type, UnknownField)
        return field_class

    def get_column_types(self, table, schema=None):
        column_types = {}
        columns = self.database.get_columns(table)
        for column in columns:
            column_types[column.name] = self._map_col(column.data_type)
        return (column_types, {})