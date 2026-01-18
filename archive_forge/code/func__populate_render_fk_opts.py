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
def _populate_render_fk_opts(constraint: ForeignKeyConstraint, opts: List[Tuple[str, str]]) -> None:
    if constraint.onupdate:
        opts.append(('onupdate', repr(constraint.onupdate)))
    if constraint.ondelete:
        opts.append(('ondelete', repr(constraint.ondelete)))
    if constraint.initially:
        opts.append(('initially', repr(constraint.initially)))
    if constraint.deferrable:
        opts.append(('deferrable', repr(constraint.deferrable)))
    if constraint.use_alter:
        opts.append(('use_alter', repr(constraint.use_alter)))
    if constraint.match:
        opts.append(('match', repr(constraint.match)))