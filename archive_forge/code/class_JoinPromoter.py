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
class JoinPromoter:
    """
    A class to abstract away join promotion problems for complex filter
    conditions.
    """

    def __init__(self, connector, num_children, negated):
        self.connector = connector
        self.negated = negated
        if self.negated:
            if connector == AND:
                self.effective_connector = OR
            else:
                self.effective_connector = AND
        else:
            self.effective_connector = self.connector
        self.num_children = num_children
        self.votes = Counter()

    def __repr__(self):
        return f'{self.__class__.__qualname__}(connector={self.connector!r}, num_children={self.num_children!r}, negated={self.negated!r})'

    def add_votes(self, votes):
        """
        Add single vote per item to self.votes. Parameter can be any
        iterable.
        """
        self.votes.update(votes)

    def update_join_types(self, query):
        """
        Change join types so that the generated query is as efficient as
        possible, but still correct. So, change as many joins as possible
        to INNER, but don't make OUTER joins INNER if that could remove
        results from the query.
        """
        to_promote = set()
        to_demote = set()
        for table, votes in self.votes.items():
            if self.effective_connector == OR and votes < self.num_children:
                to_promote.add(table)
            if self.effective_connector == AND or (self.effective_connector == OR and votes == self.num_children):
                to_demote.add(table)
        query.promote_joins(to_promote)
        query.demote_joins(to_demote)
        return to_demote