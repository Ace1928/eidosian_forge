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
@comparators.dispatch_for('table')
def _compare_table_comment(autogen_context: AutogenContext, modify_table_ops: ModifyTableOps, schema: Optional[str], tname: Union[quoted_name, str], conn_table: Optional[Table], metadata_table: Optional[Table]) -> None:
    assert autogen_context.dialect is not None
    if not autogen_context.dialect.supports_comments:
        return
    if conn_table is None or metadata_table is None:
        return
    if conn_table.comment is None and metadata_table.comment is None:
        return
    if metadata_table.comment is None and conn_table.comment is not None:
        modify_table_ops.ops.append(ops.DropTableCommentOp(tname, existing_comment=conn_table.comment, schema=schema))
    elif metadata_table.comment != conn_table.comment:
        modify_table_ops.ops.append(ops.CreateTableCommentOp(tname, metadata_table.comment, existing_comment=conn_table.comment, schema=schema))