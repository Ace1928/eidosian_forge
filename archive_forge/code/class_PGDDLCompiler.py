from the proposed insertion.   These values are specified using the
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import re
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import arraylib as _array
from . import json as _json
from . import pg_catalog
from . import ranges as _ranges
from .ext import _regconfig_fn
from .ext import aggregate_order_by
from .hstore import HSTORE
from .named_types import CreateDomainType as CreateDomainType  # noqa: F401
from .named_types import CreateEnumType as CreateEnumType  # noqa: F401
from .named_types import DOMAIN as DOMAIN  # noqa: F401
from .named_types import DropDomainType as DropDomainType  # noqa: F401
from .named_types import DropEnumType as DropEnumType  # noqa: F401
from .named_types import ENUM as ENUM  # noqa: F401
from .named_types import NamedType as NamedType  # noqa: F401
from .types import _DECIMAL_TYPES  # noqa: F401
from .types import _FLOAT_TYPES  # noqa: F401
from .types import _INT_TYPES  # noqa: F401
from .types import BIT as BIT
from .types import BYTEA as BYTEA
from .types import CIDR as CIDR
from .types import CITEXT as CITEXT
from .types import INET as INET
from .types import INTERVAL as INTERVAL
from .types import MACADDR as MACADDR
from .types import MACADDR8 as MACADDR8
from .types import MONEY as MONEY
from .types import OID as OID
from .types import PGBit as PGBit  # noqa: F401
from .types import PGCidr as PGCidr  # noqa: F401
from .types import PGInet as PGInet  # noqa: F401
from .types import PGInterval as PGInterval  # noqa: F401
from .types import PGMacAddr as PGMacAddr  # noqa: F401
from .types import PGMacAddr8 as PGMacAddr8  # noqa: F401
from .types import PGUuid as PGUuid
from .types import REGCLASS as REGCLASS
from .types import REGCONFIG as REGCONFIG  # noqa: F401
from .types import TIME as TIME
from .types import TIMESTAMP as TIMESTAMP
from .types import TSVECTOR as TSVECTOR
from ... import exc
from ... import schema
from ... import select
from ... import sql
from ... import util
from ...engine import characteristics
from ...engine import default
from ...engine import interfaces
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine import URL
from ...engine.reflection import ReflectionDefaults
from ...sql import bindparam
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.visitors import InternalTraversal
from ...types import BIGINT
from ...types import BOOLEAN
from ...types import CHAR
from ...types import DATE
from ...types import DOUBLE_PRECISION
from ...types import FLOAT
from ...types import INTEGER
from ...types import NUMERIC
from ...types import REAL
from ...types import SMALLINT
from ...types import TEXT
from ...types import UUID as UUID
from ...types import VARCHAR
from ...util.typing import TypedDict
class PGDDLCompiler(compiler.DDLCompiler):

    def get_column_specification(self, column, **kwargs):
        colspec = self.preparer.format_column(column)
        impl_type = column.type.dialect_impl(self.dialect)
        if isinstance(impl_type, sqltypes.TypeDecorator):
            impl_type = impl_type.impl
        has_identity = column.identity is not None and self.dialect.supports_identity_columns
        if column.primary_key and column is column.table._autoincrement_column and (self.dialect.supports_smallserial or not isinstance(impl_type, sqltypes.SmallInteger)) and (not has_identity) and (column.default is None or (isinstance(column.default, schema.Sequence) and column.default.optional)):
            if isinstance(impl_type, sqltypes.BigInteger):
                colspec += ' BIGSERIAL'
            elif isinstance(impl_type, sqltypes.SmallInteger):
                colspec += ' SMALLSERIAL'
            else:
                colspec += ' SERIAL'
        else:
            colspec += ' ' + self.dialect.type_compiler_instance.process(column.type, type_expression=column, identifier_preparer=self.preparer)
            default = self.get_column_default_string(column)
            if default is not None:
                colspec += ' DEFAULT ' + default
        if column.computed is not None:
            colspec += ' ' + self.process(column.computed)
        if has_identity:
            colspec += ' ' + self.process(column.identity)
        if not column.nullable and (not has_identity):
            colspec += ' NOT NULL'
        elif column.nullable and has_identity:
            colspec += ' NULL'
        return colspec

    def _define_constraint_validity(self, constraint):
        not_valid = constraint.dialect_options['postgresql']['not_valid']
        return ' NOT VALID' if not_valid else ''

    def visit_check_constraint(self, constraint, **kw):
        if constraint._type_bound:
            typ = list(constraint.columns)[0].type
            if isinstance(typ, sqltypes.ARRAY) and isinstance(typ.item_type, sqltypes.Enum) and (not typ.item_type.native_enum):
                raise exc.CompileError('PostgreSQL dialect cannot produce the CHECK constraint for ARRAY of non-native ENUM; please specify create_constraint=False on this Enum datatype.')
        text = super().visit_check_constraint(constraint)
        text += self._define_constraint_validity(constraint)
        return text

    def visit_foreign_key_constraint(self, constraint, **kw):
        text = super().visit_foreign_key_constraint(constraint)
        text += self._define_constraint_validity(constraint)
        return text

    def visit_create_enum_type(self, create, **kw):
        type_ = create.element
        return 'CREATE TYPE %s AS ENUM (%s)' % (self.preparer.format_type(type_), ', '.join((self.sql_compiler.process(sql.literal(e), literal_binds=True) for e in type_.enums)))

    def visit_drop_enum_type(self, drop, **kw):
        type_ = drop.element
        return 'DROP TYPE %s' % self.preparer.format_type(type_)

    def visit_create_domain_type(self, create, **kw):
        domain: DOMAIN = create.element
        options = []
        if domain.collation is not None:
            options.append(f'COLLATE {self.preparer.quote(domain.collation)}')
        if domain.default is not None:
            default = self.render_default_string(domain.default)
            options.append(f'DEFAULT {default}')
        if domain.constraint_name is not None:
            name = self.preparer.truncate_and_render_constraint_name(domain.constraint_name)
            options.append(f'CONSTRAINT {name}')
        if domain.not_null:
            options.append('NOT NULL')
        if domain.check is not None:
            check = self.sql_compiler.process(domain.check, include_table=False, literal_binds=True)
            options.append(f'CHECK ({check})')
        return f'CREATE DOMAIN {self.preparer.format_type(domain)} AS {self.type_compiler.process(domain.data_type)} {' '.join(options)}'

    def visit_drop_domain_type(self, drop, **kw):
        domain = drop.element
        return f'DROP DOMAIN {self.preparer.format_type(domain)}'

    def visit_create_index(self, create, **kw):
        preparer = self.preparer
        index = create.element
        self._verify_index_table(index)
        text = 'CREATE '
        if index.unique:
            text += 'UNIQUE '
        text += 'INDEX '
        if self.dialect._supports_create_index_concurrently:
            concurrently = index.dialect_options['postgresql']['concurrently']
            if concurrently:
                text += 'CONCURRENTLY '
        if create.if_not_exists:
            text += 'IF NOT EXISTS '
        text += '%s ON %s ' % (self._prepared_index_name(index, include_schema=False), preparer.format_table(index.table))
        using = index.dialect_options['postgresql']['using']
        if using:
            text += 'USING %s ' % self.preparer.validate_sql_phrase(using, IDX_USING).lower()
        ops = index.dialect_options['postgresql']['ops']
        text += '(%s)' % ', '.join([self.sql_compiler.process(expr.self_group() if not isinstance(expr, expression.ColumnClause) else expr, include_table=False, literal_binds=True) + (' ' + ops[expr.key] if hasattr(expr, 'key') and expr.key in ops else '') for expr in index.expressions])
        includeclause = index.dialect_options['postgresql']['include']
        if includeclause:
            inclusions = [index.table.c[col] if isinstance(col, str) else col for col in includeclause]
            text += ' INCLUDE (%s)' % ', '.join([preparer.quote(c.name) for c in inclusions])
        nulls_not_distinct = index.dialect_options['postgresql']['nulls_not_distinct']
        if nulls_not_distinct is True:
            text += ' NULLS NOT DISTINCT'
        elif nulls_not_distinct is False:
            text += ' NULLS DISTINCT'
        withclause = index.dialect_options['postgresql']['with']
        if withclause:
            text += ' WITH (%s)' % ', '.join(['%s = %s' % storage_parameter for storage_parameter in withclause.items()])
        tablespace_name = index.dialect_options['postgresql']['tablespace']
        if tablespace_name:
            text += ' TABLESPACE %s' % preparer.quote(tablespace_name)
        whereclause = index.dialect_options['postgresql']['where']
        if whereclause is not None:
            whereclause = coercions.expect(roles.DDLExpressionRole, whereclause)
            where_compiled = self.sql_compiler.process(whereclause, include_table=False, literal_binds=True)
            text += ' WHERE ' + where_compiled
        return text

    def define_unique_constraint_distinct(self, constraint, **kw):
        nulls_not_distinct = constraint.dialect_options['postgresql']['nulls_not_distinct']
        if nulls_not_distinct is True:
            nulls_not_distinct_param = 'NULLS NOT DISTINCT '
        elif nulls_not_distinct is False:
            nulls_not_distinct_param = 'NULLS DISTINCT '
        else:
            nulls_not_distinct_param = ''
        return nulls_not_distinct_param

    def visit_drop_index(self, drop, **kw):
        index = drop.element
        text = '\nDROP INDEX '
        if self.dialect._supports_drop_index_concurrently:
            concurrently = index.dialect_options['postgresql']['concurrently']
            if concurrently:
                text += 'CONCURRENTLY '
        if drop.if_exists:
            text += 'IF EXISTS '
        text += self._prepared_index_name(index, include_schema=True)
        return text

    def visit_exclude_constraint(self, constraint, **kw):
        text = ''
        if constraint.name is not None:
            text += 'CONSTRAINT %s ' % self.preparer.format_constraint(constraint)
        elements = []
        kw['include_table'] = False
        kw['literal_binds'] = True
        for expr, name, op in constraint._render_exprs:
            exclude_element = self.sql_compiler.process(expr, **kw) + (' ' + constraint.ops[expr.key] if hasattr(expr, 'key') and expr.key in constraint.ops else '')
            elements.append('%s WITH %s' % (exclude_element, op))
        text += 'EXCLUDE USING %s (%s)' % (self.preparer.validate_sql_phrase(constraint.using, IDX_USING).lower(), ', '.join(elements))
        if constraint.where is not None:
            text += ' WHERE (%s)' % self.sql_compiler.process(constraint.where, literal_binds=True)
        text += self.define_constraint_deferrability(constraint)
        return text

    def post_create_table(self, table):
        table_opts = []
        pg_opts = table.dialect_options['postgresql']
        inherits = pg_opts.get('inherits')
        if inherits is not None:
            if not isinstance(inherits, (list, tuple)):
                inherits = (inherits,)
            table_opts.append('\n INHERITS ( ' + ', '.join((self.preparer.quote(name) for name in inherits)) + ' )')
        if pg_opts['partition_by']:
            table_opts.append('\n PARTITION BY %s' % pg_opts['partition_by'])
        if pg_opts['using']:
            table_opts.append('\n USING %s' % pg_opts['using'])
        if pg_opts['with_oids'] is True:
            table_opts.append('\n WITH OIDS')
        elif pg_opts['with_oids'] is False:
            table_opts.append('\n WITHOUT OIDS')
        if pg_opts['on_commit']:
            on_commit_options = pg_opts['on_commit'].replace('_', ' ').upper()
            table_opts.append('\n ON COMMIT %s' % on_commit_options)
        if pg_opts['tablespace']:
            tablespace_name = pg_opts['tablespace']
            table_opts.append('\n TABLESPACE %s' % self.preparer.quote(tablespace_name))
        return ''.join(table_opts)

    def visit_computed_column(self, generated, **kw):
        if generated.persisted is False:
            raise exc.CompileError("PostrgreSQL computed columns do not support 'virtual' persistence; set the 'persisted' flag to None or True for PostgreSQL support.")
        return 'GENERATED ALWAYS AS (%s) STORED' % self.sql_compiler.process(generated.sqltext, include_table=False, literal_binds=True)

    def visit_create_sequence(self, create, **kw):
        prefix = None
        if create.element.data_type is not None:
            prefix = ' AS %s' % self.type_compiler.process(create.element.data_type)
        return super().visit_create_sequence(create, prefix=prefix, **kw)

    def _can_comment_on_constraint(self, ddl_instance):
        constraint = ddl_instance.element
        if constraint.name is None:
            raise exc.CompileError(f"Can't emit COMMENT ON for constraint {constraint!r}: it has no name")
        if constraint.table is None:
            raise exc.CompileError(f"Can't emit COMMENT ON for constraint {constraint!r}: it has no associated table")

    def visit_set_constraint_comment(self, create, **kw):
        self._can_comment_on_constraint(create)
        return 'COMMENT ON CONSTRAINT %s ON %s IS %s' % (self.preparer.format_constraint(create.element), self.preparer.format_table(create.element.table), self.sql_compiler.render_literal_value(create.element.comment, sqltypes.String()))

    def visit_drop_constraint_comment(self, drop, **kw):
        self._can_comment_on_constraint(drop)
        return 'COMMENT ON CONSTRAINT %s ON %s IS NULL' % (self.preparer.format_constraint(drop.element), self.preparer.format_table(drop.element.table))