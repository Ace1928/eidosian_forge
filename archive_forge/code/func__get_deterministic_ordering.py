import warnings
from datetime import datetime, timedelta
from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.admin import FieldListFilter
from django.contrib.admin.exceptions import (
from django.contrib.admin.options import (
from django.contrib.admin.utils import (
from django.core.exceptions import (
from django.core.paginator import InvalidPage
from django.db.models import F, Field, ManyToOneRel, OrderBy
from django.db.models.expressions import Combinable
from django.urls import reverse
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.http import urlencode
from django.utils.inspect import func_supports_parameter
from django.utils.timezone import make_aware
from django.utils.translation import gettext
def _get_deterministic_ordering(self, ordering):
    """
        Ensure a deterministic order across all database backends. Search for a
        single field or unique together set of fields providing a total
        ordering. If these are missing, augment the ordering with a descendant
        primary key.
        """
    ordering = list(ordering)
    ordering_fields = set()
    total_ordering_fields = {'pk'} | {field.attname for field in self.lookup_opts.fields if field.unique and (not field.null)}
    for part in ordering:
        field_name = None
        if isinstance(part, str):
            field_name = part.lstrip('-')
        elif isinstance(part, F):
            field_name = part.name
        elif isinstance(part, OrderBy) and isinstance(part.expression, F):
            field_name = part.expression.name
        if field_name:
            try:
                field = self.lookup_opts.get_field(field_name)
            except FieldDoesNotExist:
                continue
            if field.remote_field and field_name == field.name:
                continue
            if field.attname in total_ordering_fields:
                break
            ordering_fields.add(field.attname)
    else:
        constraint_field_names = (*self.lookup_opts.unique_together, *(constraint.fields for constraint in self.lookup_opts.total_unique_constraints))
        for field_names in constraint_field_names:
            fields = [self.lookup_opts.get_field(field_name) for field_name in field_names]
            if any((field.null for field in fields)):
                continue
            if ordering_fields.issuperset((field.attname for field in fields)):
                break
        else:
            ordering.append('-pk')
    return ordering