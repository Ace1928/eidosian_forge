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
@reflection.flexi_cache(('schema', InternalTraversal.dp_string), ('filter_names', InternalTraversal.dp_string_list), ('dblink', InternalTraversal.dp_string))
def _get_synonyms(self, connection, schema, filter_names, dblink, **kw):
    owner = self.denormalize_schema_name(schema or self.default_schema_name)
    has_filter_names, params = self._prepare_filter_names(filter_names)
    query = select(dictionary.all_synonyms.c.synonym_name, dictionary.all_synonyms.c.table_name, dictionary.all_synonyms.c.table_owner, dictionary.all_synonyms.c.db_link).where(dictionary.all_synonyms.c.owner == owner)
    if has_filter_names:
        query = query.where(dictionary.all_synonyms.c.synonym_name.in_(params['filter_names']))
    result = self._execute_reflection(connection, query, dblink, returns_long=False).mappings()
    return result.all()