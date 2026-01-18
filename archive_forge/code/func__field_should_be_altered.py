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
def _field_should_be_altered(self, old_field, new_field, ignore=None):
    if not old_field.concrete and (not new_field.concrete):
        return False
    ignore = ignore or set()
    _, old_path, old_args, old_kwargs = old_field.deconstruct()
    _, new_path, new_args, new_kwargs = new_field.deconstruct()
    for attr in ignore.union(old_field.non_db_attrs):
        old_kwargs.pop(attr, None)
    for attr in ignore.union(new_field.non_db_attrs):
        new_kwargs.pop(attr, None)
    if not new_field.many_to_many and old_field.remote_field and new_field.remote_field and (old_field.remote_field.model._meta.db_table == new_field.remote_field.model._meta.db_table):
        old_kwargs.pop('to', None)
        new_kwargs.pop('to', None)
    if old_kwargs.get('db_default') and new_kwargs.get('db_default') and (self.db_default_sql(old_field) == self.db_default_sql(new_field)):
        old_kwargs.pop('db_default')
        new_kwargs.pop('db_default')
    return self.quote_name(old_field.column) != self.quote_name(new_field.column) or (old_path, old_args, old_kwargs) != (new_path, new_args, new_kwargs)