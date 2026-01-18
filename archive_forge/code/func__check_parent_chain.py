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
def _check_parent_chain(self, instance):
    """
        Check if the field value can be fetched from a parent field already
        loaded in the instance. This can be done if the to-be fetched
        field is a primary key field.
        """
    opts = instance._meta
    link_field = opts.get_ancestor_link(self.field.model)
    if self.field.primary_key and self.field != link_field:
        return getattr(instance, link_field.attname)
    return None