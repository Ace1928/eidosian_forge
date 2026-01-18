import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain
from asgiref.sync import sync_to_async
import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
from django.db import (
from django.db.models import NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.expressions import DatabaseDefault, RawSQL
from django.db.models.fields.related import (
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import (
from django.db.models.utils import AltersData, make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
@classmethod
def _check_indexes(cls, databases):
    """Check fields, names, and conditions of indexes."""
    errors = []
    references = set()
    for index in cls._meta.indexes:
        if index.name[0] == '_' or index.name[0].isdigit():
            errors.append(checks.Error("The index name '%s' cannot start with an underscore or a number." % index.name, obj=cls, id='models.E033'))
        if len(index.name) > index.max_name_length:
            errors.append(checks.Error("The index name '%s' cannot be longer than %d characters." % (index.name, index.max_name_length), obj=cls, id='models.E034'))
        if index.contains_expressions:
            for expression in index.expressions:
                references.update((ref[0] for ref in cls._get_expr_references(expression)))
    for db in databases:
        if not router.allow_migrate_model(db, cls):
            continue
        connection = connections[db]
        if not (connection.features.supports_partial_indexes or 'supports_partial_indexes' in cls._meta.required_db_features) and any((index.condition is not None for index in cls._meta.indexes)):
            errors.append(checks.Warning('%s does not support indexes with conditions.' % connection.display_name, hint="Conditions will be ignored. Silence this warning if you don't care about it.", obj=cls, id='models.W037'))
        if not (connection.features.supports_covering_indexes or 'supports_covering_indexes' in cls._meta.required_db_features) and any((index.include for index in cls._meta.indexes)):
            errors.append(checks.Warning('%s does not support indexes with non-key columns.' % connection.display_name, hint="Non-key columns will be ignored. Silence this warning if you don't care about it.", obj=cls, id='models.W040'))
        if not (connection.features.supports_expression_indexes or 'supports_expression_indexes' in cls._meta.required_db_features) and any((index.contains_expressions for index in cls._meta.indexes)):
            errors.append(checks.Warning('%s does not support indexes on expressions.' % connection.display_name, hint="An index won't be created. Silence this warning if you don't care about it.", obj=cls, id='models.W043'))
    fields = [field for index in cls._meta.indexes for field, _ in index.fields_orders]
    fields += [include for index in cls._meta.indexes for include in index.include]
    fields += references
    errors.extend(cls._check_local_fields(fields, 'indexes'))
    return errors