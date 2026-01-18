from __future__ import annotations
import collections
import collections.abc as collections_abc
import contextlib
from enum import IntEnum
import functools
import itertools
import operator
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import base
from . import coercions
from . import crud
from . import elements
from . import functions
from . import operators
from . import roles
from . import schema
from . import selectable
from . import sqltypes
from . import util as sql_util
from ._typing import is_column_element
from ._typing import is_dml
from .base import _de_clone
from .base import _from_objects
from .base import _NONE_NAME
from .base import _SentinelDefaultCharacterization
from .base import Executable
from .base import NO_ARG
from .elements import ClauseElement
from .elements import quoted_name
from .schema import Column
from .sqltypes import TupleType
from .type_api import TypeEngine
from .visitors import prefix_anon_map
from .visitors import Visitable
from .. import exc
from .. import util
from ..util import FastIntFlag
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
class DDLCompiler(Compiled):
    is_ddl = True
    if TYPE_CHECKING:

        def __init__(self, dialect: Dialect, statement: ExecutableDDLElement, schema_translate_map: Optional[SchemaTranslateMapType]=..., render_schema_translate: bool=..., compile_kwargs: Mapping[str, Any]=...):
            ...

    @util.memoized_property
    def sql_compiler(self):
        return self.dialect.statement_compiler(self.dialect, None, schema_translate_map=self.schema_translate_map)

    @util.memoized_property
    def type_compiler(self):
        return self.dialect.type_compiler_instance

    def construct_params(self, params: Optional[_CoreSingleExecuteParams]=None, extracted_parameters: Optional[Sequence[BindParameter[Any]]]=None, escape_names: bool=True) -> Optional[_MutableCoreSingleExecuteParams]:
        return None

    def visit_ddl(self, ddl, **kwargs):
        context = ddl.context
        if isinstance(ddl.target, schema.Table):
            context = context.copy()
            preparer = self.preparer
            path = preparer.format_table_seq(ddl.target)
            if len(path) == 1:
                table, sch = (path[0], '')
            else:
                table, sch = (path[-1], path[0])
            context.setdefault('table', table)
            context.setdefault('schema', sch)
            context.setdefault('fullname', preparer.format_table(ddl.target))
        return self.sql_compiler.post_process_text(ddl.statement % context)

    def visit_create_schema(self, create, **kw):
        text = 'CREATE SCHEMA '
        if create.if_not_exists:
            text += 'IF NOT EXISTS '
        return text + self.preparer.format_schema(create.element)

    def visit_drop_schema(self, drop, **kw):
        text = 'DROP SCHEMA '
        if drop.if_exists:
            text += 'IF EXISTS '
        text += self.preparer.format_schema(drop.element)
        if drop.cascade:
            text += ' CASCADE'
        return text

    def visit_create_table(self, create, **kw):
        table = create.element
        preparer = self.preparer
        text = '\nCREATE '
        if table._prefixes:
            text += ' '.join(table._prefixes) + ' '
        text += 'TABLE '
        if create.if_not_exists:
            text += 'IF NOT EXISTS '
        text += preparer.format_table(table) + ' '
        create_table_suffix = self.create_table_suffix(table)
        if create_table_suffix:
            text += create_table_suffix + ' '
        text += '('
        separator = '\n'
        first_pk = False
        for create_column in create.columns:
            column = create_column.element
            try:
                processed = self.process(create_column, first_pk=column.primary_key and (not first_pk))
                if processed is not None:
                    text += separator
                    separator = ', \n'
                    text += '\t' + processed
                if column.primary_key:
                    first_pk = True
            except exc.CompileError as ce:
                raise exc.CompileError("(in table '%s', column '%s'): %s" % (table.description, column.name, ce.args[0])) from ce
        const = self.create_table_constraints(table, _include_foreign_key_constraints=create.include_foreign_key_constraints)
        if const:
            text += separator + '\t' + const
        text += '\n)%s\n\n' % self.post_create_table(table)
        return text

    def visit_create_column(self, create, first_pk=False, **kw):
        column = create.element
        if column.system:
            return None
        text = self.get_column_specification(column, first_pk=first_pk)
        const = ' '.join((self.process(constraint) for constraint in column.constraints))
        if const:
            text += ' ' + const
        return text

    def create_table_constraints(self, table, _include_foreign_key_constraints=None, **kw):
        constraints = []
        if table.primary_key:
            constraints.append(table.primary_key)
        all_fkcs = table.foreign_key_constraints
        if _include_foreign_key_constraints is not None:
            omit_fkcs = all_fkcs.difference(_include_foreign_key_constraints)
        else:
            omit_fkcs = set()
        constraints.extend([c for c in table._sorted_constraints if c is not table.primary_key and c not in omit_fkcs])
        return ', \n\t'.join((p for p in (self.process(constraint) for constraint in constraints if constraint._should_create_for_compiler(self) and (not self.dialect.supports_alter or not getattr(constraint, 'use_alter', False))) if p is not None))

    def visit_drop_table(self, drop, **kw):
        text = '\nDROP TABLE '
        if drop.if_exists:
            text += 'IF EXISTS '
        return text + self.preparer.format_table(drop.element)

    def visit_drop_view(self, drop, **kw):
        return '\nDROP VIEW ' + self.preparer.format_table(drop.element)

    def _verify_index_table(self, index):
        if index.table is None:
            raise exc.CompileError("Index '%s' is not associated with any table." % index.name)

    def visit_create_index(self, create, include_schema=False, include_table_schema=True, **kw):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        text = 'CREATE '
        if index.unique:
            text += 'UNIQUE '
        if index.name is None:
            raise exc.CompileError('CREATE INDEX requires that the index have a name')
        text += 'INDEX '
        if create.if_not_exists:
            text += 'IF NOT EXISTS '
        text += '%s ON %s (%s)' % (self._prepared_index_name(index, include_schema=include_schema), preparer.format_table(index.table, use_schema=include_table_schema), ', '.join((self.sql_compiler.process(expr, include_table=False, literal_binds=True) for expr in index.expressions)))
        return text

    def visit_drop_index(self, drop, **kw):
        index = drop.element
        if index.name is None:
            raise exc.CompileError('DROP INDEX requires that the index have a name')
        text = '\nDROP INDEX '
        if drop.if_exists:
            text += 'IF EXISTS '
        return text + self._prepared_index_name(index, include_schema=True)

    def _prepared_index_name(self, index, include_schema=False):
        if index.table is not None:
            effective_schema = self.preparer.schema_for_object(index.table)
        else:
            effective_schema = None
        if include_schema and effective_schema:
            schema_name = self.preparer.quote_schema(effective_schema)
        else:
            schema_name = None
        index_name = self.preparer.format_index(index)
        if schema_name:
            index_name = schema_name + '.' + index_name
        return index_name

    def visit_add_constraint(self, create, **kw):
        return 'ALTER TABLE %s ADD %s' % (self.preparer.format_table(create.element.table), self.process(create.element))

    def visit_set_table_comment(self, create, **kw):
        return 'COMMENT ON TABLE %s IS %s' % (self.preparer.format_table(create.element), self.sql_compiler.render_literal_value(create.element.comment, sqltypes.String()))

    def visit_drop_table_comment(self, drop, **kw):
        return 'COMMENT ON TABLE %s IS NULL' % self.preparer.format_table(drop.element)

    def visit_set_column_comment(self, create, **kw):
        return 'COMMENT ON COLUMN %s IS %s' % (self.preparer.format_column(create.element, use_table=True, use_schema=True), self.sql_compiler.render_literal_value(create.element.comment, sqltypes.String()))

    def visit_drop_column_comment(self, drop, **kw):
        return 'COMMENT ON COLUMN %s IS NULL' % self.preparer.format_column(drop.element, use_table=True)

    def visit_set_constraint_comment(self, create, **kw):
        raise exc.UnsupportedCompilationError(self, type(create))

    def visit_drop_constraint_comment(self, drop, **kw):
        raise exc.UnsupportedCompilationError(self, type(drop))

    def get_identity_options(self, identity_options):
        text = []
        if identity_options.increment is not None:
            text.append('INCREMENT BY %d' % identity_options.increment)
        if identity_options.start is not None:
            text.append('START WITH %d' % identity_options.start)
        if identity_options.minvalue is not None:
            text.append('MINVALUE %d' % identity_options.minvalue)
        if identity_options.maxvalue is not None:
            text.append('MAXVALUE %d' % identity_options.maxvalue)
        if identity_options.nominvalue is not None:
            text.append('NO MINVALUE')
        if identity_options.nomaxvalue is not None:
            text.append('NO MAXVALUE')
        if identity_options.cache is not None:
            text.append('CACHE %d' % identity_options.cache)
        if identity_options.cycle is not None:
            text.append('CYCLE' if identity_options.cycle else 'NO CYCLE')
        return ' '.join(text)

    def visit_create_sequence(self, create, prefix=None, **kw):
        text = 'CREATE SEQUENCE '
        if create.if_not_exists:
            text += 'IF NOT EXISTS '
        text += self.preparer.format_sequence(create.element)
        if prefix:
            text += prefix
        options = self.get_identity_options(create.element)
        if options:
            text += ' ' + options
        return text

    def visit_drop_sequence(self, drop, **kw):
        text = 'DROP SEQUENCE '
        if drop.if_exists:
            text += 'IF EXISTS '
        return text + self.preparer.format_sequence(drop.element)

    def visit_drop_constraint(self, drop, **kw):
        constraint = drop.element
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
        else:
            formatted_name = None
        if formatted_name is None:
            raise exc.CompileError("Can't emit DROP CONSTRAINT for constraint %r; it has no name" % drop.element)
        return 'ALTER TABLE %s DROP CONSTRAINT %s%s%s' % (self.preparer.format_table(drop.element.table), 'IF EXISTS ' if drop.if_exists else '', formatted_name, ' CASCADE' if drop.cascade else '')

    def get_column_specification(self, column, **kwargs):
        colspec = self.preparer.format_column(column) + ' ' + self.dialect.type_compiler_instance.process(column.type, type_expression=column)
        default = self.get_column_default_string(column)
        if default is not None:
            colspec += ' DEFAULT ' + default
        if column.computed is not None:
            colspec += ' ' + self.process(column.computed)
        if column.identity is not None and self.dialect.supports_identity_columns:
            colspec += ' ' + self.process(column.identity)
        if not column.nullable and (not column.identity or not self.dialect.supports_identity_columns):
            colspec += ' NOT NULL'
        return colspec

    def create_table_suffix(self, table):
        return ''

    def post_create_table(self, table):
        return ''

    def get_column_default_string(self, column):
        if isinstance(column.server_default, schema.DefaultClause):
            return self.render_default_string(column.server_default.arg)
        else:
            return None

    def render_default_string(self, default):
        if isinstance(default, str):
            return self.sql_compiler.render_literal_value(default, sqltypes.STRINGTYPE)
        else:
            return self.sql_compiler.process(default, literal_binds=True)

    def visit_table_or_column_check_constraint(self, constraint, **kw):
        if constraint.is_column_level:
            return self.visit_column_check_constraint(constraint)
        else:
            return self.visit_check_constraint(constraint)

    def visit_check_constraint(self, constraint, **kw):
        text = ''
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += 'CONSTRAINT %s ' % formatted_name
        text += 'CHECK (%s)' % self.sql_compiler.process(constraint.sqltext, include_table=False, literal_binds=True)
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_column_check_constraint(self, constraint, **kw):
        text = ''
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += 'CONSTRAINT %s ' % formatted_name
        text += 'CHECK (%s)' % self.sql_compiler.process(constraint.sqltext, include_table=False, literal_binds=True)
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_primary_key_constraint(self, constraint, **kw):
        if len(constraint) == 0:
            return ''
        text = ''
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += 'CONSTRAINT %s ' % formatted_name
        text += 'PRIMARY KEY '
        text += '(%s)' % ', '.join((self.preparer.quote(c.name) for c in (constraint.columns_autoinc_first if constraint._implicit_generated else constraint.columns)))
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_foreign_key_constraint(self, constraint, **kw):
        preparer = self.preparer
        text = ''
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += 'CONSTRAINT %s ' % formatted_name
        remote_table = list(constraint.elements)[0].column.table
        text += 'FOREIGN KEY(%s) REFERENCES %s (%s)' % (', '.join((preparer.quote(f.parent.name) for f in constraint.elements)), self.define_constraint_remote_table(constraint, remote_table, preparer), ', '.join((preparer.quote(f.column.name) for f in constraint.elements)))
        text += self.define_constraint_match(constraint)
        text += self.define_constraint_cascades(constraint)
        text += self.define_constraint_deferrability(constraint)
        return text

    def define_constraint_remote_table(self, constraint, table, preparer):
        """Format the remote table clause of a CREATE CONSTRAINT clause."""
        return preparer.format_table(table)

    def visit_unique_constraint(self, constraint, **kw):
        if len(constraint) == 0:
            return ''
        text = ''
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += 'CONSTRAINT %s ' % formatted_name
        text += 'UNIQUE %s(%s)' % (self.define_unique_constraint_distinct(constraint, **kw), ', '.join((self.preparer.quote(c.name) for c in constraint)))
        text += self.define_constraint_deferrability(constraint)
        return text

    def define_unique_constraint_distinct(self, constraint, **kw):
        return ''

    def define_constraint_cascades(self, constraint):
        text = ''
        if constraint.ondelete is not None:
            text += ' ON DELETE %s' % self.preparer.validate_sql_phrase(constraint.ondelete, FK_ON_DELETE)
        if constraint.onupdate is not None:
            text += ' ON UPDATE %s' % self.preparer.validate_sql_phrase(constraint.onupdate, FK_ON_UPDATE)
        return text

    def define_constraint_deferrability(self, constraint):
        text = ''
        if constraint.deferrable is not None:
            if constraint.deferrable:
                text += ' DEFERRABLE'
            else:
                text += ' NOT DEFERRABLE'
        if constraint.initially is not None:
            text += ' INITIALLY %s' % self.preparer.validate_sql_phrase(constraint.initially, FK_INITIALLY)
        return text

    def define_constraint_match(self, constraint):
        text = ''
        if constraint.match is not None:
            text += ' MATCH %s' % constraint.match
        return text

    def visit_computed_column(self, generated, **kw):
        text = 'GENERATED ALWAYS AS (%s)' % self.sql_compiler.process(generated.sqltext, include_table=False, literal_binds=True)
        if generated.persisted is True:
            text += ' STORED'
        elif generated.persisted is False:
            text += ' VIRTUAL'
        return text

    def visit_identity_column(self, identity, **kw):
        text = 'GENERATED %s AS IDENTITY' % ('ALWAYS' if identity.always else 'BY DEFAULT',)
        options = self.get_identity_options(identity)
        if options:
            text += ' (%s)' % options
        return text