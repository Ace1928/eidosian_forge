from __future__ import annotations
from io import StringIO
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from mako.pygen import PythonPrinter
from sqlalchemy import schema as sa_schema
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from sqlalchemy.sql.elements import conv
from sqlalchemy.sql.elements import quoted_name
from .. import util
from ..operations import ops
from ..util import sqla_compat
@renderers.dispatch_for(ops.DropConstraintOp)
def _drop_constraint(autogen_context: AutogenContext, op: ops.DropConstraintOp) -> str:
    prefix = _alembic_autogenerate_prefix(autogen_context)
    name = _render_gen_name(autogen_context, op.constraint_name)
    schema = _ident(op.schema) if op.schema else None
    type_ = _ident(op.constraint_type) if op.constraint_type else None
    params_strs = []
    params_strs.append(repr(name))
    if not autogen_context._has_batch:
        params_strs.append(repr(_ident(op.table_name)))
        if schema is not None:
            params_strs.append(f'schema={schema!r}')
    if type_ is not None:
        params_strs.append(f'type_={type_!r}')
    return f'{prefix}drop_constraint({', '.join(params_strs)})'