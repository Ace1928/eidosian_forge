import copy
import operator
import warnings
from itertools import chain, islice
from asgiref.sync import sync_to_async
import django
from django.conf import settings
from django.core import exceptions
from django.db import (
from django.db.models import AutoField, DateField, DateTimeField, Field, sql
from django.db.models.constants import LOOKUP_SEP, OnConflict
from django.db.models.deletion import Collector
from django.db.models.expressions import Case, F, Value, When
from django.db.models.functions import Cast, Trunc
from django.db.models.query_utils import FilteredRelation, Q
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE
from django.db.models.utils import (
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property, partition
def _prepare_for_bulk_create(self, objs):
    from django.db.models.expressions import DatabaseDefault
    connection = connections[self.db]
    for obj in objs:
        if obj.pk is None:
            obj.pk = obj._meta.pk.get_pk_value_on_save(obj)
        if not connection.features.supports_default_keyword_in_bulk_insert:
            for field in obj._meta.fields:
                if field.generated:
                    continue
                value = getattr(obj, field.attname)
                if isinstance(value, DatabaseDefault):
                    setattr(obj, field.attname, field.db_default)
        obj._prepare_related_fields_for_save(operation_name='bulk_create')