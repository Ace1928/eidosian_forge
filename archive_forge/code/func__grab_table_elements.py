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
def _grab_table_elements(self) -> None:
    schema = self.table.schema
    self.columns: Dict[str, Column[Any]] = OrderedDict()
    for c in self.table.c:
        c_copy = _copy(c, schema=schema)
        c_copy.unique = c_copy.index = False
        if isinstance(c.type, SchemaEventTarget):
            assert c_copy.type is not c.type
        self.columns[c.name] = c_copy
    self.named_constraints: Dict[str, Constraint] = {}
    self.unnamed_constraints = []
    self.col_named_constraints = {}
    self.indexes: Dict[str, Index] = {}
    self.new_indexes: Dict[str, Index] = {}
    for const in self.table.constraints:
        if _is_type_bound(const):
            continue
        elif self.reflected and isinstance(const, CheckConstraint) and (not const.name):
            pass
        elif constraint_name_string(const.name):
            self.named_constraints[const.name] = const
        else:
            self.unnamed_constraints.append(const)
    if not self.reflected:
        for col in self.table.c:
            for const in col.constraints:
                if const.name:
                    self.col_named_constraints[const.name] = (col, const)
    for idx in self.table.indexes:
        self.indexes[idx.name] = idx
    for k in self.table.kwargs:
        self.table_kwargs.setdefault(k, self.table.kwargs[k])