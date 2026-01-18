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
def _check_constraints(cls, databases):
    errors = []
    for db in databases:
        if not router.allow_migrate_model(db, cls):
            continue
        connection = connections[db]
        if not (connection.features.supports_table_check_constraints or 'supports_table_check_constraints' in cls._meta.required_db_features) and any((isinstance(constraint, CheckConstraint) for constraint in cls._meta.constraints)):
            errors.append(checks.Warning('%s does not support check constraints.' % connection.display_name, hint="A constraint won't be created. Silence this warning if you don't care about it.", obj=cls, id='models.W027'))
        if not (connection.features.supports_partial_indexes or 'supports_partial_indexes' in cls._meta.required_db_features) and any((isinstance(constraint, UniqueConstraint) and constraint.condition is not None for constraint in cls._meta.constraints)):
            errors.append(checks.Warning('%s does not support unique constraints with conditions.' % connection.display_name, hint="A constraint won't be created. Silence this warning if you don't care about it.", obj=cls, id='models.W036'))
        if not (connection.features.supports_deferrable_unique_constraints or 'supports_deferrable_unique_constraints' in cls._meta.required_db_features) and any((isinstance(constraint, UniqueConstraint) and constraint.deferrable is not None for constraint in cls._meta.constraints)):
            errors.append(checks.Warning('%s does not support deferrable unique constraints.' % connection.display_name, hint="A constraint won't be created. Silence this warning if you don't care about it.", obj=cls, id='models.W038'))
        if not (connection.features.supports_covering_indexes or 'supports_covering_indexes' in cls._meta.required_db_features) and any((isinstance(constraint, UniqueConstraint) and constraint.include for constraint in cls._meta.constraints)):
            errors.append(checks.Warning('%s does not support unique constraints with non-key columns.' % connection.display_name, hint="A constraint won't be created. Silence this warning if you don't care about it.", obj=cls, id='models.W039'))
        if not (connection.features.supports_expression_indexes or 'supports_expression_indexes' in cls._meta.required_db_features) and any((isinstance(constraint, UniqueConstraint) and constraint.contains_expressions for constraint in cls._meta.constraints)):
            errors.append(checks.Warning('%s does not support unique constraints on expressions.' % connection.display_name, hint="A constraint won't be created. Silence this warning if you don't care about it.", obj=cls, id='models.W044'))
        if not (connection.features.supports_nulls_distinct_unique_constraints or 'supports_nulls_distinct_unique_constraints' in cls._meta.required_db_features) and any((isinstance(constraint, UniqueConstraint) and constraint.nulls_distinct is not None for constraint in cls._meta.constraints)):
            errors.append(checks.Warning('%s does not support unique constraints with nulls distinct.' % connection.display_name, hint="A constraint won't be created. Silence this warning if you don't care about it.", obj=cls, id='models.W047'))
        fields = set(chain.from_iterable(((*constraint.fields, *constraint.include) for constraint in cls._meta.constraints if isinstance(constraint, UniqueConstraint))))
        references = set()
        for constraint in cls._meta.constraints:
            if isinstance(constraint, UniqueConstraint):
                if (connection.features.supports_partial_indexes or 'supports_partial_indexes' not in cls._meta.required_db_features) and isinstance(constraint.condition, Q):
                    references.update(cls._get_expr_references(constraint.condition))
                if (connection.features.supports_expression_indexes or 'supports_expression_indexes' not in cls._meta.required_db_features) and constraint.contains_expressions:
                    for expression in constraint.expressions:
                        references.update(cls._get_expr_references(expression))
            elif isinstance(constraint, CheckConstraint):
                if connection.features.supports_table_check_constraints or 'supports_table_check_constraints' not in cls._meta.required_db_features:
                    if isinstance(constraint.check, Q):
                        references.update(cls._get_expr_references(constraint.check))
                    if any((isinstance(expr, RawSQL) for expr in constraint.check.flatten())):
                        errors.append(checks.Warning(f"Check constraint {constraint.name!r} contains RawSQL() expression and won't be validated during the model full_clean().", hint="Silence this warning if you don't care about it.", obj=cls, id='models.W045'))
        for field_name, *lookups in references:
            if field_name != 'pk':
                fields.add(field_name)
            if not lookups:
                continue
            try:
                if field_name == 'pk':
                    field = cls._meta.pk
                else:
                    field = cls._meta.get_field(field_name)
                if not field.is_relation or field.many_to_many or field.one_to_many:
                    continue
            except FieldDoesNotExist:
                continue
            first_lookup = lookups[0]
            if hasattr(field, 'get_transform') and hasattr(field, 'get_lookup') and (field.get_transform(first_lookup) is None) and (field.get_lookup(first_lookup) is None):
                errors.append(checks.Error("'constraints' refers to the joined field '%s'." % LOOKUP_SEP.join([field_name] + lookups), obj=cls, id='models.E041'))
        errors.extend(cls._check_local_fields(fields, 'constraints'))
    return errors