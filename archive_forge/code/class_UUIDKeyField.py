import functools
import re
import sys
from peewee import *
from peewee import _atomic
from peewee import _manual
from peewee import ColumnMetadata  # (name, data_type, null, primary_key, table, default)
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import ForeignKeyMetadata  # (column, dest_table, dest_column, table).
from peewee import IndexMetadata
from peewee import NodeList
from playhouse.pool import _PooledPostgresqlDatabase
class UUIDKeyField(UUIDField):
    auto_increment = True

    def __init__(self, *args, **kwargs):
        if kwargs.get('constraints'):
            raise ValueError('%s cannot specify constraints.' % type(self))
        kwargs['constraints'] = [SQL('DEFAULT gen_random_uuid()')]
        kwargs.setdefault('primary_key', True)
        super(UUIDKeyField, self).__init__(*args, **kwargs)