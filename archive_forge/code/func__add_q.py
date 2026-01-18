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
def _add_q(self, q_object, used_aliases, branch_negated=False, current_negated=False, allow_joins=True, split_subq=True, check_filterable=True, summarize=False, update_join_types=True):
    """Add a Q-object to the current filter."""
    connector = q_object.connector
    current_negated ^= q_object.negated
    branch_negated = branch_negated or q_object.negated
    target_clause = WhereNode(connector=connector, negated=q_object.negated)
    joinpromoter = JoinPromoter(q_object.connector, len(q_object.children), current_negated)
    for child in q_object.children:
        child_clause, needed_inner = self.build_filter(child, can_reuse=used_aliases, branch_negated=branch_negated, current_negated=current_negated, allow_joins=allow_joins, split_subq=split_subq, check_filterable=check_filterable, summarize=summarize, update_join_types=update_join_types)
        joinpromoter.add_votes(needed_inner)
        if child_clause:
            target_clause.add(child_clause, connector)
    if update_join_types:
        needed_inner = joinpromoter.update_join_types(self)
    else:
        needed_inner = []
    return (target_clause, needed_inner)