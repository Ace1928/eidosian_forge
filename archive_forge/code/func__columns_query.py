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
def _columns_query(self, schema, has_filter_names, scope, kind):
    generated = pg_catalog.pg_attribute.c.attgenerated.label('generated') if self.server_version_info >= (12,) else sql.null().label('generated')
    if self.server_version_info >= (10,):
        identity = select(sql.func.json_build_object('always', pg_catalog.pg_attribute.c.attidentity == 'a', 'start', pg_catalog.pg_sequence.c.seqstart, 'increment', pg_catalog.pg_sequence.c.seqincrement, 'minvalue', pg_catalog.pg_sequence.c.seqmin, 'maxvalue', pg_catalog.pg_sequence.c.seqmax, 'cache', pg_catalog.pg_sequence.c.seqcache, 'cycle', pg_catalog.pg_sequence.c.seqcycle)).select_from(pg_catalog.pg_sequence).where(pg_catalog.pg_attribute.c.attidentity != '', pg_catalog.pg_sequence.c.seqrelid == sql.cast(sql.cast(pg_catalog.pg_get_serial_sequence(sql.cast(sql.cast(pg_catalog.pg_attribute.c.attrelid, REGCLASS), TEXT), pg_catalog.pg_attribute.c.attname), REGCLASS), OID)).correlate(pg_catalog.pg_attribute).scalar_subquery().label('identity_options')
    else:
        identity = sql.null().label('identity_options')
    default = select(pg_catalog.pg_get_expr(pg_catalog.pg_attrdef.c.adbin, pg_catalog.pg_attrdef.c.adrelid)).select_from(pg_catalog.pg_attrdef).where(pg_catalog.pg_attrdef.c.adrelid == pg_catalog.pg_attribute.c.attrelid, pg_catalog.pg_attrdef.c.adnum == pg_catalog.pg_attribute.c.attnum, pg_catalog.pg_attribute.c.atthasdef).correlate(pg_catalog.pg_attribute).scalar_subquery().label('default')
    relkinds = self._kind_to_relkinds(kind)
    query = select(pg_catalog.pg_attribute.c.attname.label('name'), pg_catalog.format_type(pg_catalog.pg_attribute.c.atttypid, pg_catalog.pg_attribute.c.atttypmod).label('format_type'), default, pg_catalog.pg_attribute.c.attnotnull.label('not_null'), pg_catalog.pg_class.c.relname.label('table_name'), pg_catalog.pg_description.c.description.label('comment'), generated, identity).select_from(pg_catalog.pg_class).outerjoin(pg_catalog.pg_attribute, sql.and_(pg_catalog.pg_class.c.oid == pg_catalog.pg_attribute.c.attrelid, pg_catalog.pg_attribute.c.attnum > 0, ~pg_catalog.pg_attribute.c.attisdropped)).outerjoin(pg_catalog.pg_description, sql.and_(pg_catalog.pg_description.c.objoid == pg_catalog.pg_attribute.c.attrelid, pg_catalog.pg_description.c.objsubid == pg_catalog.pg_attribute.c.attnum)).where(self._pg_class_relkind_condition(relkinds)).order_by(pg_catalog.pg_class.c.relname, pg_catalog.pg_attribute.c.attnum)
    query = self._pg_class_filter_scope_schema(query, schema, scope=scope)
    if has_filter_names:
        query = query.where(pg_catalog.pg_class.c.relname.in_(bindparam('filter_names')))
    return query