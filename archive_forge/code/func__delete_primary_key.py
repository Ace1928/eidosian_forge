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
def _delete_primary_key(self, model, strict=False):
    constraint_names = self._constraint_names(model, primary_key=True)
    if strict and len(constraint_names) != 1:
        raise ValueError('Found wrong number (%s) of PK constraints for %s' % (len(constraint_names), model._meta.db_table))
    for constraint_name in constraint_names:
        self.execute(self._delete_primary_key_sql(model, constraint_name))