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
def _get_index_rendered_expressions(idx: Index, autogen_context: AutogenContext) -> List[str]:
    return [repr(_ident(getattr(exp, 'name', None))) if isinstance(exp, sa_schema.Column) else _render_potential_expr(exp, autogen_context, is_index=True) for exp in idx.expressions]