from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import schema
from sqlalchemy import types as sqltypes
from .base import alter_table
from .base import AlterColumn
from .base import ColumnDefault
from .base import ColumnName
from .base import ColumnNullable
from .base import ColumnType
from .base import format_column_name
from .base import format_server_default
from .impl import DefaultImpl
from .. import util
from ..util import sqla_compat
from ..util.sqla_compat import _is_mariadb
from ..util.sqla_compat import _is_type_bound
from ..util.sqla_compat import compiles
@compiles(schema.DropConstraint, 'mysql', 'mariadb')
def _mysql_drop_constraint(element: DropConstraint, compiler: MySQLDDLCompiler, **kw) -> str:
    """Redefine SQLAlchemy's drop constraint to
    raise errors for invalid constraint type."""
    constraint = element.element
    if isinstance(constraint, (schema.ForeignKeyConstraint, schema.PrimaryKeyConstraint, schema.UniqueConstraint)):
        assert not kw
        return compiler.visit_drop_constraint(element)
    elif isinstance(constraint, schema.CheckConstraint):
        if _is_mariadb(compiler.dialect):
            return 'ALTER TABLE %s DROP CONSTRAINT %s' % (compiler.preparer.format_table(constraint.table), compiler.preparer.format_constraint(constraint))
        else:
            return 'ALTER TABLE %s DROP CHECK %s' % (compiler.preparer.format_table(constraint.table), compiler.preparer.format_constraint(constraint))
    else:
        raise NotImplementedError("No generic 'DROP CONSTRAINT' in MySQL - please specify constraint type")