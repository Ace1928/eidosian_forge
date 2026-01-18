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
@_constraint_renderers.dispatch_for(sa_schema.PrimaryKeyConstraint)
def _render_primary_key(constraint: PrimaryKeyConstraint, autogen_context: AutogenContext, namespace_metadata: Optional[MetaData]) -> Optional[str]:
    rendered = _user_defined_render('primary_key', constraint, autogen_context)
    if rendered is not False:
        return rendered
    if not constraint.columns:
        return None
    opts = []
    if constraint.name:
        opts.append(('name', repr(_render_gen_name(autogen_context, constraint.name))))
    return '%(prefix)sPrimaryKeyConstraint(%(args)s)' % {'prefix': _sqlalchemy_autogenerate_prefix(autogen_context), 'args': ', '.join([repr(c.name) for c in constraint.columns] + ['%s=%s' % (kwname, val) for kwname, val in opts])}