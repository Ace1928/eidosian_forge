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
def _unregister_instance_lookup(self, lookup, lookup_name=None):
    """
        Remove given lookup from instance lookups. For use in tests only as
        it's not thread-safe.
        """
    if lookup_name is None:
        lookup_name = lookup.lookup_name
    del self.instance_lookups[lookup_name]