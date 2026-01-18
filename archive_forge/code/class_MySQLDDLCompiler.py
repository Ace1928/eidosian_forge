from the proposed insertion.   These values are normally specified using
from __future__ import annotations
from array import array as _array
from collections import defaultdict
from itertools import compress
import re
from typing import cast
from . import reflection as _reflection
from .enumerated import ENUM
from .enumerated import SET
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from .reserved_words import RESERVED_WORDS_MARIADB
from .reserved_words import RESERVED_WORDS_MYSQL
from .types import _FloatType
from .types import _IntegerType
from .types import _MatchType
from .types import _NumericType
from .types import _StringType
from .types import BIGINT
from .types import BIT
from .types import CHAR
from .types import DATETIME
from .types import DECIMAL
from .types import DOUBLE
from .types import FLOAT
from .types import INTEGER
from .types import LONGBLOB
from .types import LONGTEXT
from .types import MEDIUMBLOB
from .types import MEDIUMINT
from .types import MEDIUMTEXT
from .types import NCHAR
from .types import NUMERIC
from .types import NVARCHAR
from .types import REAL
from .types import SMALLINT
from .types import TEXT
from .types import TIME
from .types import TIMESTAMP
from .types import TINYBLOB
from .types import TINYINT
from .types import TINYTEXT
from .types import VARCHAR
from .types import YEAR
from ... import exc
from ... import literal_column
from ... import log
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import functions
from ...sql import operators
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.compiler import SQLCompiler
from ...sql.schema import SchemaConst
from ...types import BINARY
from ...types import BLOB
from ...types import BOOLEAN
from ...types import DATE
from ...types import UUID
from ...types import VARBINARY
from ...util import topological
class MySQLDDLCompiler(compiler.DDLCompiler):

    def get_column_specification(self, column, **kw):
        """Builds column DDL."""
        if self.dialect.is_mariadb is True and column.computed is not None and (column._user_defined_nullable is SchemaConst.NULL_UNSPECIFIED):
            column.nullable = True
        colspec = [self.preparer.format_column(column), self.dialect.type_compiler_instance.process(column.type, type_expression=column)]
        if column.computed is not None:
            colspec.append(self.process(column.computed))
        is_timestamp = isinstance(column.type._unwrapped_dialect_impl(self.dialect), sqltypes.TIMESTAMP)
        if not column.nullable:
            colspec.append('NOT NULL')
        elif column.nullable and is_timestamp:
            colspec.append('NULL')
        comment = column.comment
        if comment is not None:
            literal = self.sql_compiler.render_literal_value(comment, sqltypes.String())
            colspec.append('COMMENT ' + literal)
        if column.table is not None and column is column.table._autoincrement_column and (column.server_default is None or isinstance(column.server_default, sa_schema.Identity)) and (not (self.dialect.supports_sequences and isinstance(column.default, sa_schema.Sequence) and (not column.default.optional))):
            colspec.append('AUTO_INCREMENT')
        else:
            default = self.get_column_default_string(column)
            if default is not None:
                colspec.append('DEFAULT ' + default)
        return ' '.join(colspec)

    def post_create_table(self, table):
        """Build table-level CREATE options like ENGINE and COLLATE."""
        table_opts = []
        opts = {k[len(self.dialect.name) + 1:].upper(): v for k, v in table.kwargs.items() if k.startswith('%s_' % self.dialect.name)}
        if table.comment is not None:
            opts['COMMENT'] = table.comment
        partition_options = ['PARTITION_BY', 'PARTITIONS', 'SUBPARTITIONS', 'SUBPARTITION_BY']
        nonpart_options = set(opts).difference(partition_options)
        part_options = set(opts).intersection(partition_options)
        for opt in topological.sort([('DEFAULT_CHARSET', 'COLLATE'), ('DEFAULT_CHARACTER_SET', 'COLLATE'), ('CHARSET', 'COLLATE'), ('CHARACTER_SET', 'COLLATE')], nonpart_options):
            arg = opts[opt]
            if opt in _reflection._options_of_type_string:
                arg = self.sql_compiler.render_literal_value(arg, sqltypes.String())
            if opt in ('DATA_DIRECTORY', 'INDEX_DIRECTORY', 'DEFAULT_CHARACTER_SET', 'CHARACTER_SET', 'DEFAULT_CHARSET', 'DEFAULT_COLLATE'):
                opt = opt.replace('_', ' ')
            joiner = '='
            if opt in ('TABLESPACE', 'DEFAULT CHARACTER SET', 'CHARACTER SET', 'COLLATE'):
                joiner = ' '
            table_opts.append(joiner.join((opt, arg)))
        for opt in topological.sort([('PARTITION_BY', 'PARTITIONS'), ('PARTITION_BY', 'SUBPARTITION_BY'), ('PARTITION_BY', 'SUBPARTITIONS'), ('PARTITIONS', 'SUBPARTITIONS'), ('PARTITIONS', 'SUBPARTITION_BY'), ('SUBPARTITION_BY', 'SUBPARTITIONS')], part_options):
            arg = opts[opt]
            if opt in _reflection._options_of_type_string:
                arg = self.sql_compiler.render_literal_value(arg, sqltypes.String())
            opt = opt.replace('_', ' ')
            joiner = ' '
            table_opts.append(joiner.join((opt, arg)))
        return ' '.join(table_opts)

    def visit_create_index(self, create, **kw):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        table = preparer.format_table(index.table)
        columns = [self.sql_compiler.process(elements.Grouping(expr) if isinstance(expr, elements.BinaryExpression) or (isinstance(expr, elements.UnaryExpression) and expr.modifier not in (operators.desc_op, operators.asc_op)) or isinstance(expr, functions.FunctionElement) else expr, include_table=False, literal_binds=True) for expr in index.expressions]
        name = self._prepared_index_name(index)
        text = 'CREATE '
        if index.unique:
            text += 'UNIQUE '
        index_prefix = index.kwargs.get('%s_prefix' % self.dialect.name, None)
        if index_prefix:
            text += index_prefix + ' '
        text += 'INDEX '
        if create.if_not_exists:
            text += 'IF NOT EXISTS '
        text += '%s ON %s ' % (name, table)
        length = index.dialect_options[self.dialect.name]['length']
        if length is not None:
            if isinstance(length, dict):
                columns = ', '.join(('%s(%d)' % (expr, length[col.name]) if col.name in length else '%s(%d)' % (expr, length[expr]) if expr in length else '%s' % expr for col, expr in zip(index.expressions, columns)))
            else:
                columns = ', '.join(('%s(%d)' % (col, length) for col in columns))
        else:
            columns = ', '.join(columns)
        text += '(%s)' % columns
        parser = index.dialect_options['mysql']['with_parser']
        if parser is not None:
            text += ' WITH PARSER %s' % (parser,)
        using = index.dialect_options['mysql']['using']
        if using is not None:
            text += ' USING %s' % preparer.quote(using)
        return text

    def visit_primary_key_constraint(self, constraint, **kw):
        text = super().visit_primary_key_constraint(constraint)
        using = constraint.dialect_options['mysql']['using']
        if using:
            text += ' USING %s' % self.preparer.quote(using)
        return text

    def visit_drop_index(self, drop, **kw):
        index = drop.element
        text = '\nDROP INDEX '
        if drop.if_exists:
            text += 'IF EXISTS '
        return text + '%s ON %s' % (self._prepared_index_name(index, include_schema=False), self.preparer.format_table(index.table))

    def visit_drop_constraint(self, drop, **kw):
        constraint = drop.element
        if isinstance(constraint, sa_schema.ForeignKeyConstraint):
            qual = 'FOREIGN KEY '
            const = self.preparer.format_constraint(constraint)
        elif isinstance(constraint, sa_schema.PrimaryKeyConstraint):
            qual = 'PRIMARY KEY '
            const = ''
        elif isinstance(constraint, sa_schema.UniqueConstraint):
            qual = 'INDEX '
            const = self.preparer.format_constraint(constraint)
        elif isinstance(constraint, sa_schema.CheckConstraint):
            if self.dialect.is_mariadb:
                qual = 'CONSTRAINT '
            else:
                qual = 'CHECK '
            const = self.preparer.format_constraint(constraint)
        else:
            qual = ''
            const = self.preparer.format_constraint(constraint)
        return 'ALTER TABLE %s DROP %s%s' % (self.preparer.format_table(constraint.table), qual, const)

    def define_constraint_match(self, constraint):
        if constraint.match is not None:
            raise exc.CompileError("MySQL ignores the 'MATCH' keyword while at the same time causes ON UPDATE/ON DELETE clauses to be ignored.")
        return ''

    def visit_set_table_comment(self, create, **kw):
        return 'ALTER TABLE %s COMMENT %s' % (self.preparer.format_table(create.element), self.sql_compiler.render_literal_value(create.element.comment, sqltypes.String()))

    def visit_drop_table_comment(self, create, **kw):
        return "ALTER TABLE %s COMMENT ''" % self.preparer.format_table(create.element)

    def visit_set_column_comment(self, create, **kw):
        return 'ALTER TABLE %s CHANGE %s %s' % (self.preparer.format_table(create.element.table), self.preparer.format_column(create.element), self.get_column_specification(create.element))