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
def _related_non_m2m_objects(old_field, new_field):
    related_fields = zip((obj for obj in _all_related_fields(old_field.model) if _is_relevant_relation(obj, old_field)), (obj for obj in _all_related_fields(new_field.model) if _is_relevant_relation(obj, new_field)))
    for old_rel, new_rel in related_fields:
        yield (old_rel, new_rel)
        yield from _related_non_m2m_objects(old_rel.remote_field, new_rel.remote_field)