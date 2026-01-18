from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from functools import wraps
import re
from . import dictionary
from .types import _OracleBoolean
from .types import _OracleDate
from .types import BFILE
from .types import BINARY_DOUBLE
from .types import BINARY_FLOAT
from .types import DATE
from .types import FLOAT
from .types import INTERVAL
from .types import LONG
from .types import NCLOB
from .types import NUMBER
from .types import NVARCHAR2  # noqa
from .types import OracleRaw  # noqa
from .types import RAW
from .types import ROWID  # noqa
from .types import TIMESTAMP
from .types import VARCHAR2  # noqa
from ... import Computed
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import default
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import and_
from ...sql import bindparam
from ...sql import compiler
from ...sql import expression
from ...sql import func
from ...sql import null
from ...sql import or_
from ...sql import select
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.visitors import InternalTraversal
from ...types import BLOB
from ...types import CHAR
from ...types import CLOB
from ...types import DOUBLE_PRECISION
from ...types import INTEGER
from ...types import NCHAR
from ...types import NVARCHAR
from ...types import REAL
from ...types import VARCHAR
def _handle_synonyms(self, fn, connection, *args, **kwargs):
    if not kwargs.get('oracle_resolve_synonyms', False):
        return fn(self, connection, *args, **kwargs)
    original_kw = kwargs.copy()
    schema = kwargs.pop('schema', None)
    result = self._get_synonyms(connection, schema=schema, filter_names=kwargs.pop('filter_names', None), dblink=kwargs.pop('dblink', None), info_cache=kwargs.get('info_cache', None))
    dblinks_owners = defaultdict(dict)
    for row in result:
        key = (row['db_link'], row['table_owner'])
        tn = self.normalize_name(row['table_name'])
        dblinks_owners[key][tn] = row['synonym_name']
    if not dblinks_owners:
        return fn(self, connection, *args, **original_kw)
    data = {}
    for (dblink, table_owner), mapping in dblinks_owners.items():
        call_kw = {**original_kw, 'schema': table_owner, 'dblink': self.normalize_name(dblink), 'filter_names': mapping.keys()}
        call_result = fn(self, connection, *args, **call_kw)
        for (_, tn), value in call_result:
            synonym_name = self.normalize_name(mapping[tn])
            data[schema, synonym_name] = value
    return data.items()