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
def add_filtered_relation(self, filtered_relation, alias):
    filtered_relation.alias = alias
    relation_lookup_parts, relation_field_parts, _ = self.solve_lookup_type(filtered_relation.relation_name)
    if relation_lookup_parts:
        raise ValueError("FilteredRelation's relation_name cannot contain lookups (got %r)." % filtered_relation.relation_name)
    for lookup in get_children_from_q(filtered_relation.condition):
        lookup_parts, lookup_field_parts, _ = self.solve_lookup_type(lookup)
        shift = 2 if not lookup_parts else 1
        lookup_field_path = lookup_field_parts[:-shift]
        for idx, lookup_field_part in enumerate(lookup_field_path):
            if len(relation_field_parts) > idx:
                if relation_field_parts[idx] != lookup_field_part:
                    raise ValueError("FilteredRelation's condition doesn't support relations outside the %r (got %r)." % (filtered_relation.relation_name, lookup))
            else:
                raise ValueError("FilteredRelation's condition doesn't support nested relations deeper than the relation_name (got %r for %r)." % (lookup, filtered_relation.relation_name))
    filtered_relation.condition = rename_prefix_from_q(filtered_relation.relation_name, alias, filtered_relation.condition)
    self._filtered_relations[filtered_relation.alias] = filtered_relation