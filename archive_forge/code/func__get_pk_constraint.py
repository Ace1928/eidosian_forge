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
def _get_pk_constraint(self, table, schema=None):
    query = 'SELECT constraint_name FROM information_schema.table_constraints WHERE table_name = %s AND table_schema = %s AND constraint_type = %s'
    cursor = self.execute_sql(query, (table, schema or 'public', 'PRIMARY KEY'))
    row = cursor.fetchone()
    return row and row[0] or None