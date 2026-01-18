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
def _render_server_default(default: Optional[Union[FetchedValue, str, TextClause, ColumnElement[Any]]], autogen_context: AutogenContext, repr_: bool=True) -> Optional[str]:
    rendered = _user_defined_render('server_default', default, autogen_context)
    if rendered is not False:
        return rendered
    if sqla_compat._server_default_is_computed(default):
        return _render_computed(cast('Computed', default), autogen_context)
    elif sqla_compat._server_default_is_identity(default):
        return _render_identity(cast('Identity', default), autogen_context)
    elif isinstance(default, sa_schema.DefaultClause):
        if isinstance(default.arg, str):
            default = default.arg
        else:
            return _render_potential_expr(default.arg, autogen_context, is_server_default=True)
    if isinstance(default, str) and repr_:
        default = repr(re.sub("^'|'$", '', default))
    return cast(str, default)