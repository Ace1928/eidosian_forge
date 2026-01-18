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
def _fk_constraint_name(self, model, field, suffix):

    def create_fk_name(*args, **kwargs):
        return self.quote_name(self._create_index_name(*args, **kwargs))
    return ForeignKeyName(model._meta.db_table, [field.column], split_identifier(field.target_field.model._meta.db_table)[1], [field.target_field.column], suffix, create_fk_name)