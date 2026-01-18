from __future__ import annotations
import contextlib
import logging
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import event
from sqlalchemy import inspect
from sqlalchemy import schema as sa_schema
from sqlalchemy import text
from sqlalchemy import types as sqltypes
from sqlalchemy.sql import expression
from sqlalchemy.sql.schema import ForeignKeyConstraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.sql.schema import UniqueConstraint
from sqlalchemy.util import OrderedSet
from .. import util
from ..ddl._autogen import is_index_sig
from ..ddl._autogen import is_uq_sig
from ..operations import ops
from ..util import sqla_compat
@comparators.dispatch_for('column')
def _compare_nullable(autogen_context: AutogenContext, alter_column_op: AlterColumnOp, schema: Optional[str], tname: Union[quoted_name, str], cname: Union[quoted_name, str], conn_col: Column[Any], metadata_col: Column[Any]) -> None:
    metadata_col_nullable = metadata_col.nullable
    conn_col_nullable = conn_col.nullable
    alter_column_op.existing_nullable = conn_col_nullable
    if conn_col_nullable is not metadata_col_nullable:
        if sqla_compat._server_default_is_computed(metadata_col.server_default, conn_col.server_default) and sqla_compat._nullability_might_be_unset(metadata_col) or sqla_compat._server_default_is_identity(metadata_col.server_default, conn_col.server_default):
            log.info("Ignoring nullable change on identity column '%s.%s'", tname, cname)
        else:
            alter_column_op.modify_nullable = metadata_col_nullable
            log.info("Detected %s on column '%s.%s'", 'NULL' if metadata_col_nullable else 'NOT NULL', tname, cname)