from collections import namedtuple
import sqlparse
from django.db import DatabaseError
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo
from django.db.models import Index
from django.utils.regex_helper import _lazy_re_compile
class FlexibleFieldLookupDict:
    base_data_types_reverse = {'bool': 'BooleanField', 'boolean': 'BooleanField', 'smallint': 'SmallIntegerField', 'smallint unsigned': 'PositiveSmallIntegerField', 'smallinteger': 'SmallIntegerField', 'int': 'IntegerField', 'integer': 'IntegerField', 'bigint': 'BigIntegerField', 'integer unsigned': 'PositiveIntegerField', 'bigint unsigned': 'PositiveBigIntegerField', 'decimal': 'DecimalField', 'real': 'FloatField', 'text': 'TextField', 'char': 'CharField', 'varchar': 'CharField', 'blob': 'BinaryField', 'date': 'DateField', 'datetime': 'DateTimeField', 'time': 'TimeField'}

    def __getitem__(self, key):
        key = key.lower().split('(', 1)[0].strip()
        return self.base_data_types_reverse[key]