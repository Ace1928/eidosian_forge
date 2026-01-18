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
def _setup_autoincrement(autogen_context: AutogenContext, alter_column_op: AlterColumnOp, schema: Optional[str], tname: Union[quoted_name, str], cname: quoted_name, conn_col: Column[Any], metadata_col: Column[Any]) -> None:
    if metadata_col.table._autoincrement_column is metadata_col:
        alter_column_op.kw['autoincrement'] = True
    elif metadata_col.autoincrement is True:
        alter_column_op.kw['autoincrement'] = True
    elif metadata_col.autoincrement is False:
        alter_column_op.kw['autoincrement'] = False