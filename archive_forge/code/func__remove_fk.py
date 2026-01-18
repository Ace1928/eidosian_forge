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
def _remove_fk(obj, compare_to):
    if autogen_context.run_object_filters(obj.const, obj.name, 'foreign_key_constraint', True, compare_to):
        modify_table_ops.ops.append(ops.DropConstraintOp.from_constraint(obj.const))
        log.info('Detected removed foreign key (%s)(%s) on table %s%s', ', '.join(obj.source_columns), ', '.join(obj.target_columns), '%s.' % obj.source_schema if obj.source_schema else '', obj.source_table)