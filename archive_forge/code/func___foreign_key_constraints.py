import datetime
import uuid
from functools import lru_cache
from django.conf import settings
from django.db import DatabaseError, NotSupportedError
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta, strip_quotes, truncate_name
from django.db.models import AutoField, Exists, ExpressionWrapper, Lookup
from django.db.models.expressions import RawSQL
from django.db.models.sql.where import WhereNode
from django.utils import timezone
from django.utils.encoding import force_bytes, force_str
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from .base import Database
from .utils import BulkInsertMapper, InsertVar, Oracle_datetime
def __foreign_key_constraints(self, table_name, recursive):
    with self.connection.cursor() as cursor:
        if recursive:
            cursor.execute("\n                    SELECT\n                        user_tables.table_name, rcons.constraint_name\n                    FROM\n                        user_tables\n                    JOIN\n                        user_constraints cons\n                        ON (user_tables.table_name = cons.table_name\n                        AND cons.constraint_type = ANY('P', 'U'))\n                    LEFT JOIN\n                        user_constraints rcons\n                        ON (user_tables.table_name = rcons.table_name\n                        AND rcons.constraint_type = 'R')\n                    START WITH user_tables.table_name = UPPER(%s)\n                    CONNECT BY\n                        NOCYCLE PRIOR cons.constraint_name = rcons.r_constraint_name\n                    GROUP BY\n                        user_tables.table_name, rcons.constraint_name\n                    HAVING user_tables.table_name != UPPER(%s)\n                    ORDER BY MAX(level) DESC\n                    ", (table_name, table_name))
        else:
            cursor.execute("\n                    SELECT\n                        cons.table_name, cons.constraint_name\n                    FROM\n                        user_constraints cons\n                    WHERE\n                        cons.constraint_type = 'R'\n                        AND cons.table_name = UPPER(%s)\n                    ", (table_name,))
        return cursor.fetchall()