from __future__ import annotations
import codecs
import datetime
import operator
import re
from typing import overload
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from . import information_schema as ischema
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import Identity
from ... import schema as sa_schema
from ... import Sequence
from ... import sql
from ... import text
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import func
from ...sql import quoted_name
from ...sql import roles
from ...sql import sqltypes
from ...sql import try_cast as try_cast  # noqa: F401
from ...sql import util as sql_util
from ...sql._typing import is_sql_compiler
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.elements import TryCast as TryCast  # noqa: F401
from ...types import BIGINT
from ...types import BINARY
from ...types import CHAR
from ...types import DATE
from ...types import DATETIME
from ...types import DECIMAL
from ...types import FLOAT
from ...types import INTEGER
from ...types import NCHAR
from ...types import NUMERIC
from ...types import NVARCHAR
from ...types import SMALLINT
from ...types import TEXT
from ...types import VARCHAR
from ...util import update_wrapper
from ...util.typing import Literal
from
from
class MSDDLCompiler(compiler.DDLCompiler):

    def get_column_specification(self, column, **kwargs):
        colspec = self.preparer.format_column(column)
        if column.computed is not None:
            colspec += ' ' + self.process(column.computed)
        else:
            colspec += ' ' + self.dialect.type_compiler_instance.process(column.type, type_expression=column)
        if column.nullable is not None:
            if not column.nullable or column.primary_key or isinstance(column.default, sa_schema.Sequence) or (column.autoincrement is True) or column.identity:
                colspec += ' NOT NULL'
            elif column.computed is None:
                colspec += ' NULL'
        if column.table is None:
            raise exc.CompileError('mssql requires Table-bound columns in order to generate DDL')
        d_opt = column.dialect_options['mssql']
        start = d_opt['identity_start']
        increment = d_opt['identity_increment']
        if start is not None or increment is not None:
            if column.identity:
                raise exc.CompileError("Cannot specify options 'mssql_identity_start' and/or 'mssql_identity_increment' while also using the 'Identity' construct.")
            util.warn_deprecated("The dialect options 'mssql_identity_start' and 'mssql_identity_increment' are deprecated. Use the 'Identity' object instead.", '1.4')
        if column.identity:
            colspec += self.process(column.identity, **kwargs)
        elif (column is column.table._autoincrement_column or column.autoincrement is True) and (not isinstance(column.default, Sequence) or column.default.optional):
            colspec += self.process(Identity(start=start, increment=increment))
        else:
            default = self.get_column_default_string(column)
            if default is not None:
                colspec += ' DEFAULT ' + default
        return colspec

    def visit_create_index(self, create, include_schema=False, **kw):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        text = 'CREATE '
        if index.unique:
            text += 'UNIQUE '
        clustered = index.dialect_options['mssql']['clustered']
        if clustered is not None:
            if clustered:
                text += 'CLUSTERED '
            else:
                text += 'NONCLUSTERED '
        columnstore = index.dialect_options['mssql']['columnstore']
        if columnstore:
            text += 'COLUMNSTORE '
        text += 'INDEX %s ON %s' % (self._prepared_index_name(index, include_schema=include_schema), preparer.format_table(index.table))
        if len(index.expressions) > 0:
            text += ' (%s)' % ', '.join((self.sql_compiler.process(expr, include_table=False, literal_binds=True) for expr in index.expressions))
        if index.dialect_options['mssql']['include']:
            inclusions = [index.table.c[col] if isinstance(col, str) else col for col in index.dialect_options['mssql']['include']]
            text += ' INCLUDE (%s)' % ', '.join([preparer.quote(c.name) for c in inclusions])
        whereclause = index.dialect_options['mssql']['where']
        if whereclause is not None:
            whereclause = coercions.expect(roles.DDLExpressionRole, whereclause)
            where_compiled = self.sql_compiler.process(whereclause, include_table=False, literal_binds=True)
            text += ' WHERE ' + where_compiled
        return text

    def visit_drop_index(self, drop, **kw):
        return '\nDROP INDEX %s ON %s' % (self._prepared_index_name(drop.element, include_schema=False), self.preparer.format_table(drop.element.table))

    def visit_primary_key_constraint(self, constraint, **kw):
        if len(constraint) == 0:
            return ''
        text = ''
        if constraint.name is not None:
            text += 'CONSTRAINT %s ' % self.preparer.format_constraint(constraint)
        text += 'PRIMARY KEY '
        clustered = constraint.dialect_options['mssql']['clustered']
        if clustered is not None:
            if clustered:
                text += 'CLUSTERED '
            else:
                text += 'NONCLUSTERED '
        text += '(%s)' % ', '.join((self.preparer.quote(c.name) for c in constraint))
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_unique_constraint(self, constraint, **kw):
        if len(constraint) == 0:
            return ''
        text = ''
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += 'CONSTRAINT %s ' % formatted_name
        text += 'UNIQUE %s' % self.define_unique_constraint_distinct(constraint, **kw)
        clustered = constraint.dialect_options['mssql']['clustered']
        if clustered is not None:
            if clustered:
                text += 'CLUSTERED '
            else:
                text += 'NONCLUSTERED '
        text += '(%s)' % ', '.join((self.preparer.quote(c.name) for c in constraint))
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_computed_column(self, generated, **kw):
        text = 'AS (%s)' % self.sql_compiler.process(generated.sqltext, include_table=False, literal_binds=True)
        if generated.persisted is True:
            text += ' PERSISTED'
        return text

    def visit_set_table_comment(self, create, **kw):
        schema = self.preparer.schema_for_object(create.element)
        schema_name = schema if schema else self.dialect.default_schema_name
        return "execute sp_addextendedproperty 'MS_Description', {}, 'schema', {}, 'table', {}".format(self.sql_compiler.render_literal_value(create.element.comment, sqltypes.NVARCHAR()), self.preparer.quote_schema(schema_name), self.preparer.format_table(create.element, use_schema=False))

    def visit_drop_table_comment(self, drop, **kw):
        schema = self.preparer.schema_for_object(drop.element)
        schema_name = schema if schema else self.dialect.default_schema_name
        return "execute sp_dropextendedproperty 'MS_Description', 'schema', {}, 'table', {}".format(self.preparer.quote_schema(schema_name), self.preparer.format_table(drop.element, use_schema=False))

    def visit_set_column_comment(self, create, **kw):
        schema = self.preparer.schema_for_object(create.element.table)
        schema_name = schema if schema else self.dialect.default_schema_name
        return "execute sp_addextendedproperty 'MS_Description', {}, 'schema', {}, 'table', {}, 'column', {}".format(self.sql_compiler.render_literal_value(create.element.comment, sqltypes.NVARCHAR()), self.preparer.quote_schema(schema_name), self.preparer.format_table(create.element.table, use_schema=False), self.preparer.format_column(create.element))

    def visit_drop_column_comment(self, drop, **kw):
        schema = self.preparer.schema_for_object(drop.element.table)
        schema_name = schema if schema else self.dialect.default_schema_name
        return "execute sp_dropextendedproperty 'MS_Description', 'schema', {}, 'table', {}, 'column', {}".format(self.preparer.quote_schema(schema_name), self.preparer.format_table(drop.element.table, use_schema=False), self.preparer.format_column(drop.element))

    def visit_create_sequence(self, create, **kw):
        prefix = None
        if create.element.data_type is not None:
            data_type = create.element.data_type
            prefix = ' AS %s' % self.type_compiler.process(data_type)
        return super().visit_create_sequence(create, prefix=prefix, **kw)

    def visit_identity_column(self, identity, **kw):
        text = ' IDENTITY'
        if identity.start is not None or identity.increment is not None:
            start = 1 if identity.start is None else identity.start
            increment = 1 if identity.increment is None else identity.increment
            text += '(%s,%s)' % (start, increment)
        return text