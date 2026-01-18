import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import FieldError
from django.db import (
from django.db.models import Q, Window, signals
from django.db.models.functions import RowNumber
from django.db.models.lookups import GreaterThan, LessThanOrEqual
from django.db.models.query import QuerySet
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData, resolve_callables
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
def _get_add_plan(self, db, source_field_name):
    """
            Return a boolean triple of the way the add should be performed.

            The first element is whether or not bulk_create(ignore_conflicts)
            can be used, the second whether or not signals must be sent, and
            the third element is whether or not the immediate bulk insertion
            with conflicts ignored can be performed.
            """
    can_ignore_conflicts = self.through._meta.auto_created is not False and connections[db].features.supports_ignore_conflicts
    must_send_signals = (self.reverse or source_field_name == self.source_field_name) and signals.m2m_changed.has_listeners(self.through)
    return (can_ignore_conflicts, must_send_signals, can_ignore_conflicts and (not must_send_signals))