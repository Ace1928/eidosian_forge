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
def _render_potential_expr(value: Any, autogen_context: AutogenContext, *, wrap_in_text: bool=True, is_server_default: bool=False, is_index: bool=False) -> str:
    if isinstance(value, sql.ClauseElement):
        if wrap_in_text:
            template = '%(prefix)stext(%(sql)r)'
        else:
            template = '%(sql)r'
        return template % {'prefix': _sqlalchemy_autogenerate_prefix(autogen_context), 'sql': autogen_context.migration_context.impl.render_ddl_sql_expr(value, is_server_default=is_server_default, is_index=is_index)}
    else:
        return repr(value)