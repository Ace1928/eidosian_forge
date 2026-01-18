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
@lru_cache()
def _domain_query(self, schema):
    con_sq = select(pg_catalog.pg_constraint.c.contypid, sql.func.array_agg(pg_catalog.pg_get_constraintdef(pg_catalog.pg_constraint.c.oid, True)).label('condefs'), sql.func.array_agg(pg_catalog.pg_constraint.c.conname.cast(TEXT)).label('connames')).where(pg_catalog.pg_constraint.c.contypid != 0).group_by(pg_catalog.pg_constraint.c.contypid).subquery('domain_constraints')
    query = select(pg_catalog.pg_type.c.typname.label('name'), pg_catalog.format_type(pg_catalog.pg_type.c.typbasetype, pg_catalog.pg_type.c.typtypmod).label('attype'), (~pg_catalog.pg_type.c.typnotnull).label('nullable'), pg_catalog.pg_type.c.typdefault.label('default'), pg_catalog.pg_type_is_visible(pg_catalog.pg_type.c.oid).label('visible'), pg_catalog.pg_namespace.c.nspname.label('schema'), con_sq.c.condefs, con_sq.c.connames, pg_catalog.pg_collation.c.collname).join(pg_catalog.pg_namespace, pg_catalog.pg_namespace.c.oid == pg_catalog.pg_type.c.typnamespace).outerjoin(pg_catalog.pg_collation, pg_catalog.pg_type.c.typcollation == pg_catalog.pg_collation.c.oid).outerjoin(con_sq, pg_catalog.pg_type.c.oid == con_sq.c.contypid).where(pg_catalog.pg_type.c.typtype == 'd').order_by(pg_catalog.pg_namespace.c.nspname, pg_catalog.pg_type.c.typname)
    return self._pg_type_filter_schema(query, schema)