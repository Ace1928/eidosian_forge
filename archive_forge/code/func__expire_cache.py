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
def _expire_cache(self, forward=True, reverse=True):
    if forward:
        for cache_key in self.FORWARD_PROPERTIES:
            if cache_key in self.__dict__:
                delattr(self, cache_key)
    if reverse and (not self.abstract):
        for cache_key in self.REVERSE_PROPERTIES:
            if cache_key in self.__dict__:
                delattr(self, cache_key)
    self._get_fields_cache = {}