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
@renderers.dispatch_for(ops.ModifyTableOps)
def _render_modify_table(autogen_context: AutogenContext, op: ModifyTableOps) -> List[str]:
    opts = autogen_context.opts
    render_as_batch = opts.get('render_as_batch', False)
    if op.ops:
        lines = []
        if render_as_batch:
            with autogen_context._within_batch():
                lines.append('with op.batch_alter_table(%r, schema=%r) as batch_op:' % (op.table_name, op.schema))
                for t_op in op.ops:
                    t_lines = render_op(autogen_context, t_op)
                    lines.extend(t_lines)
                lines.append('')
        else:
            for t_op in op.ops:
                t_lines = render_op(autogen_context, t_op)
                lines.extend(t_lines)
        return lines
    else:
        return []