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
def _delete_check_sql(self, model, name):
    if not self.connection.features.supports_table_check_constraints:
        return None
    return self._delete_constraint_sql(self.sql_delete_check, model, name)