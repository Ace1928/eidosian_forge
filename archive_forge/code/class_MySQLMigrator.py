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
class MySQLMigrator(SchemaMigrator):
    explicit_create_foreign_key = True
    explicit_delete_foreign_key = True

    def _alter_column(self, ctx, table, column):
        return self._alter_table(ctx, table).literal(' MODIFY ').sql(Entity(column))

    @operation
    def rename_table(self, old_name, new_name):
        return self.make_context().literal('RENAME TABLE ').sql(Entity(old_name)).literal(' TO ').sql(Entity(new_name))

    def _get_column_definition(self, table, column_name):
        cursor = self.database.execute_sql('DESCRIBE `%s`;' % table)
        rows = cursor.fetchall()
        for row in rows:
            column = MySQLColumn(*row)
            if column.name == column_name:
                return column
        return False

    def get_foreign_key_constraint(self, table, column_name):
        cursor = self.database.execute_sql('SELECT constraint_name FROM information_schema.key_column_usage WHERE table_schema = DATABASE() AND table_name = %s AND column_name = %s AND referenced_table_name IS NOT NULL AND referenced_column_name IS NOT NULL;', (table, column_name))
        result = cursor.fetchone()
        if not result:
            raise AttributeError('Unable to find foreign key constraint for "%s" on table "%s".' % (table, column_name))
        return result[0]

    @operation
    def drop_foreign_key_constraint(self, table, column_name):
        fk_constraint = self.get_foreign_key_constraint(table, column_name)
        return self._alter_table(self.make_context(), table).literal(' DROP FOREIGN KEY ').sql(Entity(fk_constraint))

    def add_inline_fk_sql(self, ctx, field):
        pass

    @operation
    def add_not_null(self, table, column):
        column_def = self._get_column_definition(table, column)
        add_not_null = self._alter_table(self.make_context(), table).literal(' MODIFY ').sql(column_def.sql(is_null=False))
        fk_objects = dict(((fk.column, fk) for fk in self.database.get_foreign_keys(table)))
        if column not in fk_objects:
            return add_not_null
        fk_metadata = fk_objects[column]
        return (self.drop_foreign_key_constraint(table, column), add_not_null, self.add_foreign_key_constraint(table, column, fk_metadata.dest_table, fk_metadata.dest_column))

    @operation
    def drop_not_null(self, table, column):
        column = self._get_column_definition(table, column)
        if column.is_pk:
            raise ValueError('Primary keys can not be null')
        return self._alter_table(self.make_context(), table).literal(' MODIFY ').sql(column.sql(is_null=True))

    @operation
    def rename_column(self, table, old_name, new_name):
        fk_objects = dict(((fk.column, fk) for fk in self.database.get_foreign_keys(table)))
        is_foreign_key = old_name in fk_objects
        column = self._get_column_definition(table, old_name)
        rename_ctx = self._alter_table(self.make_context(), table).literal(' CHANGE ').sql(Entity(old_name)).literal(' ').sql(column.sql(column_name=new_name))
        if is_foreign_key:
            fk_metadata = fk_objects[old_name]
            return [self.drop_foreign_key_constraint(table, old_name), rename_ctx, self.add_foreign_key_constraint(table, new_name, fk_metadata.dest_table, fk_metadata.dest_column)]
        else:
            return rename_ctx

    @operation
    def alter_column_type(self, table, column, field, cast=None):
        if cast is not None:
            raise ValueError('alter_column_type() does not support cast with MySQL.')
        ctx = self.make_context()
        return self._alter_table(ctx, table).literal(' MODIFY ').sql(Entity(column)).literal(' ').sql(field.ddl(ctx))

    @operation
    def drop_index(self, table, index_name):
        return self.make_context().literal('DROP INDEX ').sql(Entity(index_name)).literal(' ON ').sql(Entity(table))