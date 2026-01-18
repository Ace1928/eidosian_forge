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
def _create_fk_sql(self, model, field, suffix):
    table = Table(model._meta.db_table, self.quote_name)
    name = self._fk_constraint_name(model, field, suffix)
    column = Columns(model._meta.db_table, [field.column], self.quote_name)
    to_table = Table(field.target_field.model._meta.db_table, self.quote_name)
    to_column = Columns(field.target_field.model._meta.db_table, [field.target_field.column], self.quote_name)
    deferrable = self.connection.ops.deferrable_sql()
    return Statement(self.sql_create_fk, table=table, name=name, column=column, to_table=to_table, to_column=to_column, deferrable=deferrable)