import logging
import operator
from datetime import datetime
from django.conf import settings
from django.db.backends.ddl_references import (
from django.db.backends.utils import names_digest, split_identifier, truncate_name
from django.db.models import NOT_PROVIDED, Deferrable, Index
from django.db.models.sql import Query
from django.db.transaction import TransactionManagementError, atomic
from django.utils import timezone
def _create_unique_sql(self, model, fields, name=None, condition=None, deferrable=None, include=None, opclasses=None, expressions=None, nulls_distinct=None):
    if not self._unique_supported(condition=condition, deferrable=deferrable, include=include, expressions=expressions, nulls_distinct=nulls_distinct):
        return None
    compiler = Query(model, alias_cols=False).get_compiler(connection=self.connection)
    table = model._meta.db_table
    columns = [field.column for field in fields]
    if name is None:
        name = self._unique_constraint_name(table, columns, quote=True)
    else:
        name = self.quote_name(name)
    if condition or include or opclasses or expressions:
        sql = self.sql_create_unique_index
    else:
        sql = self.sql_create_unique
    if columns:
        columns = self._index_columns(table, columns, col_suffixes=(), opclasses=opclasses)
    else:
        columns = Expressions(table, expressions, compiler, self.quote_value)
    return Statement(sql, table=Table(table, self.quote_name), name=name, columns=columns, condition=self._index_condition_sql(condition), deferrable=self._deferrable_constraint_sql(deferrable), include=self._index_include_sql(model, include), nulls_distinct=self._unique_index_nulls_distinct_sql(nulls_distinct))