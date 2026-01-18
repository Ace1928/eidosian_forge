from __future__ import annotations
import re
from typing import Any
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import cast
from sqlalchemy import JSON
from sqlalchemy import schema
from sqlalchemy import sql
from .base import alter_table
from .base import format_table_name
from .base import RenameTable
from .impl import DefaultImpl
from .. import util
from ..util.sqla_compat import compiles
def _guess_if_default_is_unparenthesized_sql_expr(self, expr: Optional[str]) -> bool:
    """Determine if a server default is a SQL expression or a constant.

        There are too many assertions that expect server defaults to round-trip
        identically without parenthesis added so we will add parens only in
        very specific cases.

        """
    if not expr:
        return False
    elif re.match('^[0-9\\.]$', expr):
        return False
    elif re.match("^'.+'$", expr):
        return False
    elif re.match('^\\(.+\\)$', expr):
        return False
    else:
        return True