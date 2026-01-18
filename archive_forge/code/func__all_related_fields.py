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
def _all_related_fields(model):
    return sorted(model._meta._get_fields(forward=False, reverse=True, include_hidden=True, include_parents=False), key=operator.attrgetter('name'))