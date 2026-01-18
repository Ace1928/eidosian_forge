import copy
import datetime
import functools
import inspect
from collections import defaultdict
from decimal import Decimal
from types import NoneType
from uuid import UUID
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError, connection
from django.db.models import fields
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import Q
from django.utils.deconstruct import deconstructible
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
class ResolvedOuterRef(F):
    """
    An object that contains a reference to an outer query.

    In this case, the reference to the outer query has been resolved because
    the inner query has been used as a subquery.
    """
    contains_aggregate = False
    contains_over_clause = False

    def as_sql(self, *args, **kwargs):
        raise ValueError('This queryset contains a reference to an outer query and may only be used in a subquery.')

    def resolve_expression(self, *args, **kwargs):
        col = super().resolve_expression(*args, **kwargs)
        if col.contains_over_clause:
            raise NotSupportedError(f'Referencing outer query window expression is not supported: {self.name}.')
        col.possibly_multivalued = LOOKUP_SEP in self.name
        return col

    def relabeled_clone(self, relabels):
        return self

    def get_group_by_cols(self):
        return []