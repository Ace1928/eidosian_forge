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
def _setup_referent(self, metadata: MetaData, constraint: ForeignKeyConstraint) -> None:
    spec = constraint.elements[0]._get_colspec()
    parts = spec.split('.')
    tname = parts[-2]
    if len(parts) == 3:
        referent_schema = parts[0]
    else:
        referent_schema = None
    if tname != self.temp_table_name:
        key = sql_schema._get_table_key(tname, referent_schema)

        def colspec(elem: Any):
            return elem._get_colspec()
        if key in metadata.tables:
            t = metadata.tables[key]
            for elem in constraint.elements:
                colname = colspec(elem).split('.')[-1]
                if colname not in t.c:
                    t.append_column(Column(colname, sqltypes.NULLTYPE))
        else:
            Table(tname, metadata, *[Column(n, sqltypes.NULLTYPE) for n in [colspec(elem).split('.')[-1] for elem in constraint.elements]], schema=referent_schema)