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
@comparators.dispatch_for('schema')
def _autogen_for_tables(autogen_context: AutogenContext, upgrade_ops: UpgradeOps, schemas: Union[Set[None], Set[Optional[str]]]) -> None:
    inspector = autogen_context.inspector
    conn_table_names: Set[Tuple[Optional[str], str]] = set()
    version_table_schema = autogen_context.migration_context.version_table_schema
    version_table = autogen_context.migration_context.version_table
    for schema_name in schemas:
        tables = set(inspector.get_table_names(schema=schema_name))
        if schema_name == version_table_schema:
            tables = tables.difference([autogen_context.migration_context.version_table])
        conn_table_names.update(((schema_name, tname) for tname in tables if autogen_context.run_name_filters(tname, 'table', {'schema_name': schema_name})))
    metadata_table_names = OrderedSet([(table.schema, table.name) for table in autogen_context.sorted_tables]).difference([(version_table_schema, version_table)])
    _compare_tables(conn_table_names, metadata_table_names, inspector, upgrade_ops, autogen_context)