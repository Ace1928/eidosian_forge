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
def add_extra(self, select, select_params, where, params, tables, order_by):
    """
        Add data to the various extra_* attributes for user-created additions
        to the query.
        """
    if select:
        select_pairs = {}
        if select_params:
            param_iter = iter(select_params)
        else:
            param_iter = iter([])
        for name, entry in select.items():
            self.check_alias(name)
            entry = str(entry)
            entry_params = []
            pos = entry.find('%s')
            while pos != -1:
                if pos == 0 or entry[pos - 1] != '%':
                    entry_params.append(next(param_iter))
                pos = entry.find('%s', pos + 2)
            select_pairs[name] = (entry, entry_params)
        self.extra.update(select_pairs)
    if where or params:
        self.where.add(ExtraWhere(where, params), AND)
    if tables:
        self.extra_tables += tuple(tables)
    if order_by:
        self.extra_order_by = order_by