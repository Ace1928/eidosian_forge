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
def _uq_constraint(constraint: UniqueConstraint, autogen_context: AutogenContext, alter: bool) -> str:
    opts: List[Tuple[str, Any]] = []
    has_batch = autogen_context._has_batch
    if constraint.deferrable:
        opts.append(('deferrable', str(constraint.deferrable)))
    if constraint.initially:
        opts.append(('initially', str(constraint.initially)))
    if not has_batch and alter and constraint.table.schema:
        opts.append(('schema', _ident(constraint.table.schema)))
    if not alter and constraint.name:
        opts.append(('name', _render_gen_name(autogen_context, constraint.name)))
    dialect_options = _render_dialect_kwargs_items(autogen_context, constraint)
    if alter:
        args = [repr(_render_gen_name(autogen_context, constraint.name))]
        if not has_batch:
            args += [repr(_ident(constraint.table.name))]
        args.append(repr([_ident(col.name) for col in constraint.columns]))
        args.extend(['%s=%r' % (k, v) for k, v in opts])
        args.extend(dialect_options)
        return '%(prefix)screate_unique_constraint(%(args)s)' % {'prefix': _alembic_autogenerate_prefix(autogen_context), 'args': ', '.join(args)}
    else:
        args = [repr(_ident(col.name)) for col in constraint.columns]
        args.extend(['%s=%r' % (k, v) for k, v in opts])
        args.extend(dialect_options)
        return '%(prefix)sUniqueConstraint(%(args)s)' % {'prefix': _sqlalchemy_autogenerate_prefix(autogen_context), 'args': ', '.join(args)}