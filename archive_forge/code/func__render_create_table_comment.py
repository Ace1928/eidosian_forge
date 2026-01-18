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
@renderers.dispatch_for(ops.CreateTableCommentOp)
def _render_create_table_comment(autogen_context: AutogenContext, op: ops.CreateTableCommentOp) -> str:
    if autogen_context._has_batch:
        templ = '{prefix}create_table_comment(\n{indent}{comment},\n{indent}existing_comment={existing}\n)'
    else:
        templ = "{prefix}create_table_comment(\n{indent}'{tname}',\n{indent}{comment},\n{indent}existing_comment={existing},\n{indent}schema={schema}\n)"
    return templ.format(prefix=_alembic_autogenerate_prefix(autogen_context), tname=op.table_name, comment='%r' % op.comment if op.comment is not None else None, existing='%r' % op.existing_comment if op.existing_comment is not None else None, schema="'%s'" % op.schema if op.schema is not None else None, indent='    ')