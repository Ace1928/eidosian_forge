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
@_constraint_renderers.dispatch_for(sa_schema.UniqueConstraint)
def _render_unique_constraint(constraint: UniqueConstraint, autogen_context: AutogenContext, namespace_metadata: Optional[MetaData]) -> str:
    rendered = _user_defined_render('unique', constraint, autogen_context)
    if rendered is not False:
        return rendered
    return _uq_constraint(constraint, autogen_context, False)