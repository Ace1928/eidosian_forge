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
def _delete_unique_sql(self, model, name, condition=None, deferrable=None, include=None, opclasses=None, expressions=None, nulls_distinct=None):
    if not self._unique_supported(condition=condition, deferrable=deferrable, include=include, expressions=expressions, nulls_distinct=nulls_distinct):
        return None
    if condition or include or opclasses or expressions:
        sql = self.sql_delete_index
    else:
        sql = self.sql_delete_unique
    return self._delete_constraint_sql(sql, model, name)