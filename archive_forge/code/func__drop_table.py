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
@renderers.dispatch_for(ops.DropTableOp)
def _drop_table(autogen_context: AutogenContext, op: ops.DropTableOp) -> str:
    text = '%(prefix)sdrop_table(%(tname)r' % {'prefix': _alembic_autogenerate_prefix(autogen_context), 'tname': _ident(op.table_name)}
    if op.schema:
        text += ', schema=%r' % _ident(op.schema)
    text += ')'
    return text