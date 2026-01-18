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
def _render_column(column: Column[Any], autogen_context: AutogenContext) -> str:
    rendered = _user_defined_render('column', column, autogen_context)
    if rendered is not False:
        return rendered
    args: List[str] = []
    opts: List[Tuple[str, Any]] = []
    if column.server_default:
        rendered = _render_server_default(column.server_default, autogen_context)
        if rendered:
            if _should_render_server_default_positionally(column.server_default):
                args.append(rendered)
            else:
                opts.append(('server_default', rendered))
    if column.autoincrement is not None and column.autoincrement != sqla_compat.AUTOINCREMENT_DEFAULT:
        opts.append(('autoincrement', column.autoincrement))
    if column.nullable is not None:
        opts.append(('nullable', column.nullable))
    if column.system:
        opts.append(('system', column.system))
    comment = column.comment
    if comment:
        opts.append(('comment', '%r' % comment))
    return '%(prefix)sColumn(%(name)r, %(type)s, %(args)s%(kwargs)s)' % {'prefix': _sqlalchemy_autogenerate_prefix(autogen_context), 'name': _ident(column.name), 'type': _repr_type(column.type, autogen_context), 'args': ', '.join([str(arg) for arg in args]) + ', ' if args else '', 'kwargs': ', '.join(['%s=%s' % (kwname, val) for kwname, val in opts] + ['%s=%s' % (key, _render_potential_expr(val, autogen_context)) for key, val in sqla_compat._column_kwargs(column).items()])}