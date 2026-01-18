from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
class SQLiteDDLCompiler(compiler.DDLCompiler):

    def get_column_specification(self, column, **kwargs):
        coltype = self.dialect.type_compiler_instance.process(column.type, type_expression=column)
        colspec = self.preparer.format_column(column) + ' ' + coltype
        default = self.get_column_default_string(column)
        if default is not None:
            if isinstance(column.server_default.arg, ColumnElement):
                default = '(' + default + ')'
            colspec += ' DEFAULT ' + default
        if not column.nullable:
            colspec += ' NOT NULL'
            on_conflict_clause = column.dialect_options['sqlite']['on_conflict_not_null']
            if on_conflict_clause is not None:
                colspec += ' ON CONFLICT ' + on_conflict_clause
        if column.primary_key:
            if column.autoincrement is True and len(column.table.primary_key.columns) != 1:
                raise exc.CompileError('SQLite does not support autoincrement for composite primary keys')
            if column.table.dialect_options['sqlite']['autoincrement'] and len(column.table.primary_key.columns) == 1 and issubclass(column.type._type_affinity, sqltypes.Integer) and (not column.foreign_keys):
                colspec += ' PRIMARY KEY'
                on_conflict_clause = column.dialect_options['sqlite']['on_conflict_primary_key']
                if on_conflict_clause is not None:
                    colspec += ' ON CONFLICT ' + on_conflict_clause
                colspec += ' AUTOINCREMENT'
        if column.computed is not None:
            colspec += ' ' + self.process(column.computed)
        return colspec

    def visit_primary_key_constraint(self, constraint, **kw):
        if len(constraint.columns) == 1:
            c = list(constraint)[0]
            if c.primary_key and c.table.dialect_options['sqlite']['autoincrement'] and issubclass(c.type._type_affinity, sqltypes.Integer) and (not c.foreign_keys):
                return None
        text = super().visit_primary_key_constraint(constraint)
        on_conflict_clause = constraint.dialect_options['sqlite']['on_conflict']
        if on_conflict_clause is None and len(constraint.columns) == 1:
            on_conflict_clause = list(constraint)[0].dialect_options['sqlite']['on_conflict_primary_key']
        if on_conflict_clause is not None:
            text += ' ON CONFLICT ' + on_conflict_clause
        return text

    def visit_unique_constraint(self, constraint, **kw):
        text = super().visit_unique_constraint(constraint)
        on_conflict_clause = constraint.dialect_options['sqlite']['on_conflict']
        if on_conflict_clause is None and len(constraint.columns) == 1:
            col1 = list(constraint)[0]
            if isinstance(col1, schema.SchemaItem):
                on_conflict_clause = list(constraint)[0].dialect_options['sqlite']['on_conflict_unique']
        if on_conflict_clause is not None:
            text += ' ON CONFLICT ' + on_conflict_clause
        return text

    def visit_check_constraint(self, constraint, **kw):
        text = super().visit_check_constraint(constraint)
        on_conflict_clause = constraint.dialect_options['sqlite']['on_conflict']
        if on_conflict_clause is not None:
            text += ' ON CONFLICT ' + on_conflict_clause
        return text

    def visit_column_check_constraint(self, constraint, **kw):
        text = super().visit_column_check_constraint(constraint)
        if constraint.dialect_options['sqlite']['on_conflict'] is not None:
            raise exc.CompileError('SQLite does not support on conflict clause for column check constraint')
        return text

    def visit_foreign_key_constraint(self, constraint, **kw):
        local_table = constraint.elements[0].parent.table
        remote_table = constraint.elements[0].column.table
        if local_table.schema != remote_table.schema:
            return None
        else:
            return super().visit_foreign_key_constraint(constraint)

    def define_constraint_remote_table(self, constraint, table, preparer):
        """Format the remote table clause of a CREATE CONSTRAINT clause."""
        return preparer.format_table(table, use_schema=False)

    def visit_create_index(self, create, include_schema=False, include_table_schema=True, **kw):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        text = 'CREATE '
        if index.unique:
            text += 'UNIQUE '
        text += 'INDEX '
        if create.if_not_exists:
            text += 'IF NOT EXISTS '
        text += '%s ON %s (%s)' % (self._prepared_index_name(index, include_schema=True), preparer.format_table(index.table, use_schema=False), ', '.join((self.sql_compiler.process(expr, include_table=False, literal_binds=True) for expr in index.expressions)))
        whereclause = index.dialect_options['sqlite']['where']
        if whereclause is not None:
            where_compiled = self.sql_compiler.process(whereclause, include_table=False, literal_binds=True)
            text += ' WHERE ' + where_compiled
        return text

    def post_create_table(self, table):
        if table.dialect_options['sqlite']['with_rowid'] is False:
            return '\n WITHOUT ROWID'
        return ''