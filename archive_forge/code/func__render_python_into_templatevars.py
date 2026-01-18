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
def _render_python_into_templatevars(autogen_context: AutogenContext, migration_script: MigrationScript, template_args: Dict[str, Union[str, Config]]) -> None:
    imports = autogen_context.imports
    for upgrade_ops, downgrade_ops in zip(migration_script.upgrade_ops_list, migration_script.downgrade_ops_list):
        template_args[upgrade_ops.upgrade_token] = _indent(_render_cmd_body(upgrade_ops, autogen_context))
        template_args[downgrade_ops.downgrade_token] = _indent(_render_cmd_body(downgrade_ops, autogen_context))
    template_args['imports'] = '\n'.join(sorted(imports))