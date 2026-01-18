import warnings
from enum import Enum
from types import NoneType
from django.core.exceptions import FieldError, ValidationError
from django.db import connections
from django.db.models.expressions import Exists, ExpressionList, F, OrderBy
from django.db.models.indexes import IndexExpression
from django.db.models.lookups import Exact
from django.db.models.query_utils import Q
from django.db.models.sql.query import Query
from django.db.utils import DEFAULT_DB_ALIAS
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.translation import gettext_lazy as _
def _get_check_sql(self, model, schema_editor):
    query = Query(model=model, alias_cols=False)
    where = query.build_where(self.check)
    compiler = query.get_compiler(connection=schema_editor.connection)
    sql, params = where.as_sql(compiler, schema_editor.connection)
    return sql % tuple((schema_editor.quote_value(p) for p in params))