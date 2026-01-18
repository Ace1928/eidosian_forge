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
def _reflect_type(self, format_type: Optional[str], domains: dict[str, ReflectedDomain], enums: dict[str, ReflectedEnum], type_description: str) -> sqltypes.TypeEngine[Any]:
    """
        Attempts to reconstruct a column type defined in ischema_names based
        on the information available in the format_type.

        If the `format_type` cannot be associated with a known `ischema_names`,
        it is treated as a reference to a known PostgreSQL named `ENUM` or
        `DOMAIN` type.
        """
    type_description = type_description or 'unknown type'
    if format_type is None:
        util.warn('PostgreSQL format_type() returned NULL for %s' % type_description)
        return sqltypes.NULLTYPE
    attype_args_match = self._format_type_args_pattern.search(format_type)
    if attype_args_match and attype_args_match.group(1):
        attype_args = self._format_type_args_delim.split(attype_args_match.group(1))
    else:
        attype_args = ()
    match_array_dim = self._format_array_spec_pattern.search(format_type)
    array_dim = len(match_array_dim.group(1) or '') // 2
    attype = self._format_type_args_pattern.sub('', format_type)
    attype = self._format_array_spec_pattern.sub('', attype)
    schema_type = self.ischema_names.get(attype.lower(), None)
    args, kwargs = ((), {})
    if attype == 'numeric':
        if len(attype_args) == 2:
            precision, scale = map(int, attype_args)
            args = (precision, scale)
    elif attype == 'double precision':
        args = (53,)
    elif attype == 'integer':
        args = ()
    elif attype in ('timestamp with time zone', 'time with time zone'):
        kwargs['timezone'] = True
        if len(attype_args) == 1:
            kwargs['precision'] = int(attype_args[0])
    elif attype in ('timestamp without time zone', 'time without time zone', 'time'):
        kwargs['timezone'] = False
        if len(attype_args) == 1:
            kwargs['precision'] = int(attype_args[0])
    elif attype == 'bit varying':
        kwargs['varying'] = True
        if len(attype_args) == 1:
            charlen = int(attype_args[0])
            args = (charlen,)
    elif attype.startswith('interval'):
        schema_type = INTERVAL
        field_match = re.match('interval (.+)', attype)
        if field_match:
            kwargs['fields'] = field_match.group(1)
        if len(attype_args) == 1:
            kwargs['precision'] = int(attype_args[0])
    else:
        enum_or_domain_key = tuple(util.quoted_token_parser(attype))
        if enum_or_domain_key in enums:
            schema_type = ENUM
            enum = enums[enum_or_domain_key]
            args = tuple(enum['labels'])
            kwargs['name'] = enum['name']
            if not enum['visible']:
                kwargs['schema'] = enum['schema']
            args = tuple(enum['labels'])
        elif enum_or_domain_key in domains:
            schema_type = DOMAIN
            domain = domains[enum_or_domain_key]
            data_type = self._reflect_type(domain['type'], domains, enums, type_description="DOMAIN '%s'" % domain['name'])
            args = (domain['name'], data_type)
            kwargs['collation'] = domain['collation']
            kwargs['default'] = domain['default']
            kwargs['not_null'] = not domain['nullable']
            kwargs['create_type'] = False
            if domain['constraints']:
                check_constraint = domain['constraints'][0]
                kwargs['constraint_name'] = check_constraint['name']
                kwargs['check'] = check_constraint['check']
            if not domain['visible']:
                kwargs['schema'] = domain['schema']
        else:
            try:
                charlen = int(attype_args[0])
                args = (charlen, *attype_args[1:])
            except (ValueError, IndexError):
                args = attype_args
    if not schema_type:
        util.warn("Did not recognize type '%s' of %s" % (attype, type_description))
        return sqltypes.NULLTYPE
    data_type = schema_type(*args, **kwargs)
    if array_dim >= 1:
        data_type = _array.ARRAY(data_type)
    return data_type