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
def _user_defined_render(type_: str, object_: Any, autogen_context: AutogenContext) -> Union[str, Literal[False]]:
    if 'render_item' in autogen_context.opts:
        render = autogen_context.opts['render_item']
        if render:
            rendered = render(type_, object_, autogen_context)
            if rendered is not False:
                return rendered
    return False