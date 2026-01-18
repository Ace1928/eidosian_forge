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
def _filter_prefetch_queryset(queryset, field_name, instances):
    predicate = Q(**{f'{field_name}__in': instances})
    db = queryset._db or DEFAULT_DB_ALIAS
    if queryset.query.is_sliced:
        if not connections[db].features.supports_over_clause:
            raise NotSupportedError('Prefetching from a limited queryset is only supported on backends that support window functions.')
        low_mark, high_mark = (queryset.query.low_mark, queryset.query.high_mark)
        order_by = [expr for expr, _ in queryset.query.get_compiler(using=db).get_order_by()]
        window = Window(RowNumber(), partition_by=field_name, order_by=order_by)
        predicate &= GreaterThan(window, low_mark)
        if high_mark is not None:
            predicate &= LessThanOrEqual(window, high_mark)
        queryset.query.clear_limits()
    return queryset.filter(predicate)