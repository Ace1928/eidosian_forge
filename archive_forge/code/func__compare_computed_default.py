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
def _compare_computed_default(autogen_context: AutogenContext, alter_column_op: AlterColumnOp, schema: Optional[str], tname: str, cname: str, conn_col: Column[Any], metadata_col: Column[Any]) -> None:
    rendered_metadata_default = str(cast(sa_schema.Computed, metadata_col.server_default).sqltext.compile(dialect=autogen_context.dialect, compile_kwargs={'literal_binds': True}))
    rendered_metadata_default = _normalize_computed_default(rendered_metadata_default)
    if isinstance(conn_col.server_default, sa_schema.Computed):
        rendered_conn_default = str(conn_col.server_default.sqltext.compile(dialect=autogen_context.dialect, compile_kwargs={'literal_binds': True}))
        if rendered_conn_default is None:
            rendered_conn_default = ''
        else:
            rendered_conn_default = _normalize_computed_default(rendered_conn_default)
    else:
        rendered_conn_default = ''
    if rendered_metadata_default != rendered_conn_default:
        _warn_computed_not_supported(tname, cname)