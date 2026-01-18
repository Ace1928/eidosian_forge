import functools
import inspect
import logging
from collections import namedtuple
from django.core.exceptions import FieldError
from django.db import DEFAULT_DB_ALIAS, DatabaseError, connections
from django.db.models.constants import LOOKUP_SEP
from django.utils import tree
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
def _unregister_class_lookup(cls, lookup, lookup_name=None):
    """
        Remove given lookup from cls lookups. For use in tests only as it's
        not thread-safe.
        """
    if lookup_name is None:
        lookup_name = lookup.lookup_name
    del cls.class_lookups[lookup_name]
    cls._clear_cached_class_lookups()