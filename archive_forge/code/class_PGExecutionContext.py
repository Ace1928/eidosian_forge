from the proposed insertion.   These values are specified using the
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import re
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import arraylib as _array
from . import json as _json
from . import pg_catalog
from . import ranges as _ranges
from .ext import _regconfig_fn
from .ext import aggregate_order_by
from .hstore import HSTORE
from .named_types import CreateDomainType as CreateDomainType  # noqa: F401
from .named_types import CreateEnumType as CreateEnumType  # noqa: F401
from .named_types import DOMAIN as DOMAIN  # noqa: F401
from .named_types import DropDomainType as DropDomainType  # noqa: F401
from .named_types import DropEnumType as DropEnumType  # noqa: F401
from .named_types import ENUM as ENUM  # noqa: F401
from .named_types import NamedType as NamedType  # noqa: F401
from .types import _DECIMAL_TYPES  # noqa: F401
from .types import _FLOAT_TYPES  # noqa: F401
from .types import _INT_TYPES  # noqa: F401
from .types import BIT as BIT
from .types import BYTEA as BYTEA
from .types import CIDR as CIDR
from .types import CITEXT as CITEXT
from .types import INET as INET
from .types import INTERVAL as INTERVAL
from .types import MACADDR as MACADDR
from .types import MACADDR8 as MACADDR8
from .types import MONEY as MONEY
from .types import OID as OID
from .types import PGBit as PGBit  # noqa: F401
from .types import PGCidr as PGCidr  # noqa: F401
from .types import PGInet as PGInet  # noqa: F401
from .types import PGInterval as PGInterval  # noqa: F401
from .types import PGMacAddr as PGMacAddr  # noqa: F401
from .types import PGMacAddr8 as PGMacAddr8  # noqa: F401
from .types import PGUuid as PGUuid
from .types import REGCLASS as REGCLASS
from .types import REGCONFIG as REGCONFIG  # noqa: F401
from .types import TIME as TIME
from .types import TIMESTAMP as TIMESTAMP
from .types import TSVECTOR as TSVECTOR
from ... import exc
from ... import schema
from ... import select
from ... import sql
from ... import util
from ...engine import characteristics
from ...engine import default
from ...engine import interfaces
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine import URL
from ...engine.reflection import ReflectionDefaults
from ...sql import bindparam
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.visitors import InternalTraversal
from ...types import BIGINT
from ...types import BOOLEAN
from ...types import CHAR
from ...types import DATE
from ...types import DOUBLE_PRECISION
from ...types import FLOAT
from ...types import INTEGER
from ...types import NUMERIC
from ...types import REAL
from ...types import SMALLINT
from ...types import TEXT
from ...types import UUID as UUID
from ...types import VARCHAR
from ...util.typing import TypedDict
class PGExecutionContext(default.DefaultExecutionContext):

    def fire_sequence(self, seq, type_):
        return self._execute_scalar("select nextval('%s')" % self.identifier_preparer.format_sequence(seq), type_)

    def get_insert_default(self, column):
        if column.primary_key and column is column.table._autoincrement_column:
            if column.server_default and column.server_default.has_argument:
                return self._execute_scalar('select %s' % column.server_default.arg, column.type)
            elif column.default is None or (column.default.is_sequence and column.default.optional):
                try:
                    seq_name = column._postgresql_seq_name
                except AttributeError:
                    tab = column.table.name
                    col = column.name
                    tab = tab[0:29 + max(0, 29 - len(col))]
                    col = col[0:29 + max(0, 29 - len(tab))]
                    name = '%s_%s_seq' % (tab, col)
                    column._postgresql_seq_name = seq_name = name
                if column.table is not None:
                    effective_schema = self.connection.schema_for_object(column.table)
                else:
                    effective_schema = None
                if effective_schema is not None:
                    exc = 'select nextval(\'"%s"."%s"\')' % (effective_schema, seq_name)
                else:
                    exc = 'select nextval(\'"%s"\')' % (seq_name,)
                return self._execute_scalar(exc, column.type)
        return super().get_insert_default(column)