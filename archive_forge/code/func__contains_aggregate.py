import operator
from functools import reduce
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.models.expressions import Case, When
from django.db.models.functions import Mod
from django.db.models.lookups import Exact
from django.utils import tree
from django.utils.functional import cached_property
@classmethod
def _contains_aggregate(cls, obj):
    if isinstance(obj, tree.Node):
        return any((cls._contains_aggregate(c) for c in obj.children))
    return obj.contains_aggregate