from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import schema as sql_schema
from sqlalchemy import Table
from sqlalchemy import types as sqltypes
from sqlalchemy.sql.schema import SchemaEventTarget
from sqlalchemy.util import OrderedDict
from sqlalchemy.util import topological
from ..util import exc
from ..util.sqla_compat import _columns_for_constraint
from ..util.sqla_compat import _copy
from ..util.sqla_compat import _copy_expression
from ..util.sqla_compat import _ensure_scope_for_ddl
from ..util.sqla_compat import _fk_is_self_referential
from ..util.sqla_compat import _idx_table_bound_expressions
from ..util.sqla_compat import _insert_inline
from ..util.sqla_compat import _is_type_bound
from ..util.sqla_compat import _remove_column_from_collection
from ..util.sqla_compat import _resolve_for_variant
from ..util.sqla_compat import _select
from ..util.sqla_compat import constraint_name_defined
from ..util.sqla_compat import constraint_name_string
class BatchOperationsImpl:

    def __init__(self, operations, table_name, schema, recreate, copy_from, table_args, table_kwargs, reflect_args, reflect_kwargs, naming_convention, partial_reordering):
        self.operations = operations
        self.table_name = table_name
        self.schema = schema
        if recreate not in ('auto', 'always', 'never'):
            raise ValueError("recreate may be one of 'auto', 'always', or 'never'.")
        self.recreate = recreate
        self.copy_from = copy_from
        self.table_args = table_args
        self.table_kwargs = dict(table_kwargs)
        self.reflect_args = reflect_args
        self.reflect_kwargs = dict(reflect_kwargs)
        self.reflect_kwargs.setdefault('listeners', list(self.reflect_kwargs.get('listeners', ())))
        self.reflect_kwargs['listeners'].append(('column_reflect', operations.impl.autogen_column_reflect))
        self.naming_convention = naming_convention
        self.partial_reordering = partial_reordering
        self.batch = []

    @property
    def dialect(self) -> Dialect:
        return self.operations.impl.dialect

    @property
    def impl(self) -> DefaultImpl:
        return self.operations.impl

    def _should_recreate(self) -> bool:
        if self.recreate == 'auto':
            return self.operations.impl.requires_recreate_in_batch(self)
        elif self.recreate == 'always':
            return True
        else:
            return False

    def flush(self) -> None:
        should_recreate = self._should_recreate()
        with _ensure_scope_for_ddl(self.impl.connection):
            if not should_recreate:
                for opname, arg, kw in self.batch:
                    fn = getattr(self.operations.impl, opname)
                    fn(*arg, **kw)
            else:
                if self.naming_convention:
                    m1 = MetaData(naming_convention=self.naming_convention)
                else:
                    m1 = MetaData()
                if self.copy_from is not None:
                    existing_table = self.copy_from
                    reflected = False
                else:
                    if self.operations.migration_context.as_sql:
                        raise exc.CommandError(f'This operation cannot proceed in --sql mode; batch mode with dialect {self.operations.migration_context.dialect.name} requires a live database connection with which to reflect the table "{self.table_name}". To generate a batch SQL migration script using table "move and copy", a complete Table object should be passed to the "copy_from" argument of the batch_alter_table() method so that table reflection can be skipped.')
                    existing_table = Table(self.table_name, m1, *self.reflect_args, schema=self.schema, autoload_with=self.operations.get_bind(), **self.reflect_kwargs)
                    reflected = True
                batch_impl = ApplyBatchImpl(self.impl, existing_table, self.table_args, self.table_kwargs, reflected, partial_reordering=self.partial_reordering)
                for opname, arg, kw in self.batch:
                    fn = getattr(batch_impl, opname)
                    fn(*arg, **kw)
                batch_impl._create(self.impl)

    def alter_column(self, *arg, **kw) -> None:
        self.batch.append(('alter_column', arg, kw))

    def add_column(self, *arg, **kw) -> None:
        if ('insert_before' in kw or 'insert_after' in kw) and (not self._should_recreate()):
            raise exc.CommandError("Can't specify insert_before or insert_after when using ALTER; please specify recreate='always'")
        self.batch.append(('add_column', arg, kw))

    def drop_column(self, *arg, **kw) -> None:
        self.batch.append(('drop_column', arg, kw))

    def add_constraint(self, const: Constraint) -> None:
        self.batch.append(('add_constraint', (const,), {}))

    def drop_constraint(self, const: Constraint) -> None:
        self.batch.append(('drop_constraint', (const,), {}))

    def rename_table(self, *arg, **kw):
        self.batch.append(('rename_table', arg, kw))

    def create_index(self, idx: Index, **kw: Any) -> None:
        self.batch.append(('create_index', (idx,), kw))

    def drop_index(self, idx: Index, **kw: Any) -> None:
        self.batch.append(('drop_index', (idx,), kw))

    def create_table_comment(self, table):
        self.batch.append(('create_table_comment', (table,), {}))

    def drop_table_comment(self, table):
        self.batch.append(('drop_table_comment', (table,), {}))

    def create_table(self, table):
        raise NotImplementedError("Can't create table in batch mode")

    def drop_table(self, table):
        raise NotImplementedError("Can't drop table in batch mode")

    def create_column_comment(self, column):
        self.batch.append(('create_column_comment', (column,), {}))