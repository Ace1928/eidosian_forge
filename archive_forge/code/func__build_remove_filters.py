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
def _build_remove_filters(self, removed_vals):
    filters = Q.create([(self.source_field_name, self.related_val)])
    removed_vals_filters = not isinstance(removed_vals, QuerySet) or removed_vals._has_filters()
    if removed_vals_filters:
        filters &= Q.create([(f'{self.target_field_name}__in', removed_vals)])
    if self.symmetrical:
        symmetrical_filters = Q.create([(self.target_field_name, self.related_val)])
        if removed_vals_filters:
            symmetrical_filters &= Q.create([(f'{self.source_field_name}__in', removed_vals)])
        filters |= symmetrical_filters
    return filters