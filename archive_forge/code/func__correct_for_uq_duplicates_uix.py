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
def _correct_for_uq_duplicates_uix(conn_unique_constraints, conn_indexes, metadata_unique_constraints, metadata_indexes, dialect, impl):
    metadata_cons_names = [(sqla_compat._get_constraint_final_name(cons, dialect), cons) for cons in metadata_unique_constraints]
    metadata_uq_names = {name for name, cons in metadata_cons_names if name is not None}
    unnamed_metadata_uqs = {impl._create_metadata_constraint_sig(cons).unnamed for name, cons in metadata_cons_names if name is None}
    metadata_ix_names = {sqla_compat._get_constraint_final_name(cons, dialect) for cons in metadata_indexes if cons.unique}
    conn_ix_names = {cons.name: cons for cons in conn_indexes if cons.unique}
    uqs_dupe_indexes = {cons.name: cons for cons in conn_unique_constraints if cons.info['duplicates_index']}
    for overlap in uqs_dupe_indexes:
        if overlap not in metadata_uq_names:
            if impl._create_reflected_constraint_sig(uqs_dupe_indexes[overlap]).unnamed not in unnamed_metadata_uqs:
                conn_unique_constraints.discard(uqs_dupe_indexes[overlap])
        elif overlap not in metadata_ix_names:
            conn_indexes.discard(conn_ix_names[overlap])