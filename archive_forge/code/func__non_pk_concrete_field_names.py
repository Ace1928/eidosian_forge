import bisect
import copy
import inspect
import warnings
from collections import defaultdict
from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db import connections
from django.db.models import AutoField, Manager, OrderWrt, UniqueConstraint
from django.db.models.query_utils import PathInfo
from django.utils.datastructures import ImmutableList, OrderedSet
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.text import camel_case_to_spaces, format_lazy
from django.utils.translation import override
@cached_property
def _non_pk_concrete_field_names(self):
    """
        Return a set of the non-pk concrete field names defined on the model.
        """
    names = []
    for field in self.concrete_fields:
        if not field.primary_key:
            names.append(field.name)
            if field.name != field.attname:
                names.append(field.attname)
    return frozenset(names)