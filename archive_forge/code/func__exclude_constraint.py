from __future__ import annotations
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import Numeric
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.dialects.postgresql import BIGINT
from sqlalchemy.dialects.postgresql import ExcludeConstraint
from sqlalchemy.dialects.postgresql import INTEGER
from sqlalchemy.schema import CreateIndex
from sqlalchemy.sql.elements import ColumnClause
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.types import NULLTYPE
from .base import alter_column
from .base import alter_table
from .base import AlterColumn
from .base import ColumnComment
from .base import format_column_name
from .base import format_table_name
from .base import format_type
from .base import IdentityColumnDefault
from .base import RenameTable
from .impl import ComparisonResult
from .impl import DefaultImpl
from .. import util
from ..autogenerate import render
from ..operations import ops
from ..operations import schemaobj
from ..operations.base import BatchOperations
from ..operations.base import Operations
from ..util import sqla_compat
from ..util.sqla_compat import compiles
def _exclude_constraint(constraint: ExcludeConstraint, autogen_context: AutogenContext, alter: bool) -> str:
    opts: List[Tuple[str, Union[quoted_name, str, _f_name, None]]] = []
    has_batch = autogen_context._has_batch
    if constraint.deferrable:
        opts.append(('deferrable', str(constraint.deferrable)))
    if constraint.initially:
        opts.append(('initially', str(constraint.initially)))
    if constraint.using:
        opts.append(('using', str(constraint.using)))
    if not has_batch and alter and constraint.table.schema:
        opts.append(('schema', render._ident(constraint.table.schema)))
    if not alter and constraint.name:
        opts.append(('name', render._render_gen_name(autogen_context, constraint.name)))

    def do_expr_where_opts():
        args = ['(%s, %r)' % (_render_potential_column(sqltext, autogen_context), opstring) for sqltext, name, opstring in constraint._render_exprs]
        if constraint.where is not None:
            args.append('where=%s' % render._render_potential_expr(constraint.where, autogen_context))
        args.extend(['%s=%r' % (k, v) for k, v in opts])
        return args
    if alter:
        args = [repr(render._render_gen_name(autogen_context, constraint.name))]
        if not has_batch:
            args += [repr(render._ident(constraint.table.name))]
        args.extend(do_expr_where_opts())
        return '%(prefix)screate_exclude_constraint(%(args)s)' % {'prefix': render._alembic_autogenerate_prefix(autogen_context), 'args': ', '.join(args)}
    else:
        args = do_expr_where_opts()
        return '%(prefix)sExcludeConstraint(%(args)s)' % {'prefix': _postgresql_autogenerate_prefix(autogen_context), 'args': ', '.join(args)}