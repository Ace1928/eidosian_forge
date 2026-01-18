import copy
import difflib
import functools
import sys
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import (
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import BaseTable, Empty, Join, MultiJoin
from django.db.models.sql.where import AND, OR, ExtraWhere, NothingNode, WhereNode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from django.utils.tree import Node
def _get_only_select_mask(self, opts, mask, select_mask=None):
    if select_mask is None:
        select_mask = {}
    select_mask[opts.pk] = {}
    for field_name, field_mask in mask.items():
        field = opts.get_field(field_name)
        if field in opts.related_objects:
            field_key = field.field
        else:
            field_key = field
        field_select_mask = select_mask.setdefault(field_key, {})
        if field_mask:
            if not field.is_relation:
                raise FieldError(next(iter(field_mask)))
            related_model = field.remote_field.model._meta.concrete_model
            self._get_only_select_mask(related_model._meta, field_mask, field_select_mask)
    return select_mask