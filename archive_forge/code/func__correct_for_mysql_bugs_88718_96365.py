from the proposed insertion.   These values are normally specified using
from __future__ import annotations
from array import array as _array
from collections import defaultdict
from itertools import compress
import re
from typing import cast
from . import reflection as _reflection
from .enumerated import ENUM
from .enumerated import SET
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from .reserved_words import RESERVED_WORDS_MARIADB
from .reserved_words import RESERVED_WORDS_MYSQL
from .types import _FloatType
from .types import _IntegerType
from .types import _MatchType
from .types import _NumericType
from .types import _StringType
from .types import BIGINT
from .types import BIT
from .types import CHAR
from .types import DATETIME
from .types import DECIMAL
from .types import DOUBLE
from .types import FLOAT
from .types import INTEGER
from .types import LONGBLOB
from .types import LONGTEXT
from .types import MEDIUMBLOB
from .types import MEDIUMINT
from .types import MEDIUMTEXT
from .types import NCHAR
from .types import NUMERIC
from .types import NVARCHAR
from .types import REAL
from .types import SMALLINT
from .types import TEXT
from .types import TIME
from .types import TIMESTAMP
from .types import TINYBLOB
from .types import TINYINT
from .types import TINYTEXT
from .types import VARCHAR
from .types import YEAR
from ... import exc
from ... import literal_column
from ... import log
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import functions
from ...sql import operators
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.compiler import SQLCompiler
from ...sql.schema import SchemaConst
from ...types import BINARY
from ...types import BLOB
from ...types import BOOLEAN
from ...types import DATE
from ...types import UUID
from ...types import VARBINARY
from ...util import topological
def _correct_for_mysql_bugs_88718_96365(self, fkeys, connection):
    if self._casing in (1, 2):

        def lower(s):
            return s.lower()
    else:

        def lower(s):
            return s
    default_schema_name = connection.dialect.default_schema_name
    col_tuples = [(lower(rec['referred_schema'] or default_schema_name), lower(rec['referred_table']), col_name) for rec in fkeys for col_name in rec['referred_columns']]
    if col_tuples:
        correct_for_wrong_fk_case = connection.execute(sql.text('\n                    select table_schema, table_name, column_name\n                    from information_schema.columns\n                    where (table_schema, table_name, lower(column_name)) in\n                    :table_data;\n                ').bindparams(sql.bindparam('table_data', expanding=True)), dict(table_data=col_tuples))
        d = defaultdict(dict)
        for schema, tname, cname in correct_for_wrong_fk_case:
            d[lower(schema), lower(tname)]['SCHEMANAME'] = schema
            d[lower(schema), lower(tname)]['TABLENAME'] = tname
            d[lower(schema), lower(tname)][cname.lower()] = cname
        for fkey in fkeys:
            rec = d[lower(fkey['referred_schema'] or default_schema_name), lower(fkey['referred_table'])]
            fkey['referred_table'] = rec['TABLENAME']
            if fkey['referred_schema'] is not None:
                fkey['referred_schema'] = rec['SCHEMANAME']
            fkey['referred_columns'] = [rec[col.lower()] for col in fkey['referred_columns']]