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
def _make_foreign_key(params: Dict[str, Any], conn_table: Table) -> ForeignKeyConstraint:
    tname = params['referred_table']
    if params['referred_schema']:
        tname = '%s.%s' % (params['referred_schema'], tname)
    options = params.get('options', {})
    const = sa_schema.ForeignKeyConstraint([conn_table.c[cname] for cname in params['constrained_columns']], ['%s.%s' % (tname, n) for n in params['referred_columns']], onupdate=options.get('onupdate'), ondelete=options.get('ondelete'), deferrable=options.get('deferrable'), initially=options.get('initially'), name=params['name'])
    conn_table.append_constraint(const)
    return const