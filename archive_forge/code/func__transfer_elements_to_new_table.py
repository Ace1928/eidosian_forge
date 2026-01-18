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
def _transfer_elements_to_new_table(self) -> None:
    assert self.new_table is None, 'Can only create new table once'
    m = MetaData()
    schema = self.table.schema
    if self.partial_reordering or self.add_col_ordering:
        self._adjust_self_columns_for_partial_reordering()
    self.new_table = new_table = Table(self.temp_table_name, m, *list(self.columns.values()) + list(self.table_args), schema=schema, **self.table_kwargs)
    for const in list(self.named_constraints.values()) + self.unnamed_constraints:
        const_columns = {c.key for c in _columns_for_constraint(const)}
        if not const_columns.issubset(self.column_transfers):
            continue
        const_copy: Constraint
        if isinstance(const, ForeignKeyConstraint):
            if _fk_is_self_referential(const):
                const_copy = _copy(const, schema=schema, target_table=self.table)
            else:
                const_copy = _copy(const, schema=schema)
        else:
            const_copy = _copy(const, schema=schema, target_table=new_table)
        if isinstance(const, ForeignKeyConstraint):
            self._setup_referent(m, const)
        new_table.append_constraint(const_copy)