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
def _render_gen_name(autogen_context: AutogenContext, name: sqla_compat._ConstraintName) -> Optional[Union[quoted_name, str, _f_name]]:
    if isinstance(name, conv):
        return _f_name(_alembic_autogenerate_prefix(autogen_context), name)
    else:
        return sqla_compat.constraint_name_or_none(name)