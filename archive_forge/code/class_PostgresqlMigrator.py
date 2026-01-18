from collections import namedtuple
import functools
import hashlib
import re
from peewee import *
from peewee import CommaNodeList
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import callable_
from peewee import sort_models
from peewee import sqlite3
from peewee import _truncate_constraint_name
class PostgresqlMigrator(SchemaMigrator):

    def _primary_key_columns(self, tbl):
        query = "\n            SELECT pg_attribute.attname\n            FROM pg_index, pg_class, pg_attribute\n            WHERE\n                pg_class.oid = '%s'::regclass AND\n                indrelid = pg_class.oid AND\n                pg_attribute.attrelid = pg_class.oid AND\n                pg_attribute.attnum = any(pg_index.indkey) AND\n                indisprimary;\n        "
        cursor = self.database.execute_sql(query % tbl)
        return [row[0] for row in cursor.fetchall()]

    @operation
    def set_search_path(self, schema_name):
        return self.make_context().literal('SET search_path TO %s' % schema_name)

    @operation
    def rename_table(self, old_name, new_name):
        pk_names = self._primary_key_columns(old_name)
        ParentClass = super(PostgresqlMigrator, self)
        operations = [ParentClass.rename_table(old_name, new_name, with_context=True)]
        if len(pk_names) == 1:
            seq_name = '%s_%s_seq' % (old_name, pk_names[0])
            query = '\n                SELECT 1\n                FROM information_schema.sequences\n                WHERE LOWER(sequence_name) = LOWER(%s)\n            '
            cursor = self.database.execute_sql(query, (seq_name,))
            if bool(cursor.fetchone()):
                new_seq_name = '%s_%s_seq' % (new_name, pk_names[0])
                operations.append(ParentClass.rename_table(seq_name, new_seq_name))
        return operations