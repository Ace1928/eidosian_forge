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
def _reverse_one_to_one_field_names(self):
    """
        Return a set of reverse one to one field names pointing to the current
        model.
        """
    return frozenset((field.name for field in self.related_objects if field.one_to_one))