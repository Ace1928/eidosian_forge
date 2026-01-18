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
class CockroachDBMetadata(PostgresqlMetadata):
    column_map = PostgresqlMetadata.column_map.copy()
    column_map[20] = IntegerField
    array_types = PostgresqlMetadata.array_types.copy()
    array_types[1016] = IntegerField
    extension_import = 'from playhouse.cockroachdb import *'

    def __init__(self, database):
        Metadata.__init__(self, database)
        self.requires_extension = True
        if postgres_ext is not None:
            cursor = self.execute('select oid, typname, format_type(oid, NULL) from pg_type;')
            results = cursor.fetchall()
            for oid, typname, formatted_type in results:
                if typname == 'jsonb':
                    self.column_map[oid] = postgres_ext.BinaryJSONField
            for oid in self.array_types:
                self.column_map[oid] = postgres_ext.ArrayField