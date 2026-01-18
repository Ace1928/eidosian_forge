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
@renderers.dispatch_for(ops.AddColumnOp)
def _add_column(autogen_context: AutogenContext, op: ops.AddColumnOp) -> str:
    schema, tname, column = (op.schema, op.table_name, op.column)
    if autogen_context._has_batch:
        template = '%(prefix)sadd_column(%(column)s)'
    else:
        template = '%(prefix)sadd_column(%(tname)r, %(column)s'
        if schema:
            template += ', schema=%(schema)r'
        template += ')'
    text = template % {'prefix': _alembic_autogenerate_prefix(autogen_context), 'tname': tname, 'column': _render_column(column, autogen_context), 'schema': schema}
    return text