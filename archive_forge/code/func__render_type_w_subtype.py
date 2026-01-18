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
def _render_type_w_subtype(type_: TypeEngine, autogen_context: AutogenContext, attrname: str, regexp: str, prefix: Optional[str]=None) -> Union[Optional[str], Literal[False]]:
    outer_repr = repr(type_)
    inner_type = getattr(type_, attrname, None)
    if inner_type is None:
        return False
    inner_repr = repr(inner_type)
    inner_repr = re.sub('([\\(\\)])', '\\\\\\1', inner_repr)
    sub_type = _repr_type(getattr(type_, attrname), autogen_context)
    outer_type = re.sub(regexp + inner_repr, '\\1%s' % sub_type, outer_repr)
    if prefix:
        return '%s%s' % (prefix, outer_type)
    mod = type(type_).__module__
    if mod.startswith('sqlalchemy.dialects'):
        match = re.match('sqlalchemy\\.dialects\\.(\\w+)', mod)
        assert match is not None
        dname = match.group(1)
        return '%s.%s' % (dname, outer_type)
    elif mod.startswith('sqlalchemy'):
        prefix = _sqlalchemy_autogenerate_prefix(autogen_context)
        return '%s%s' % (prefix, outer_type)
    else:
        return None