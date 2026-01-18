import operator
from functools import reduce
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.models.expressions import Case, When
from django.db.models.functions import Mod
from django.db.models.lookups import Exact
from django.utils import tree
from django.utils.functional import cached_property
class NothingNode:
    """A node that matches nothing."""
    contains_aggregate = False
    contains_over_clause = False

    def as_sql(self, compiler=None, connection=None):
        raise EmptyResultSet