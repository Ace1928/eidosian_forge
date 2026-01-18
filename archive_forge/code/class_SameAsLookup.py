from django.contrib.gis.db.models.fields import BaseSpatialField
from django.contrib.gis.measure import Distance
from django.db import NotSupportedError
from django.db.models import Expression, Lookup, Transform
from django.db.models.sql.query import Query
from django.utils.regex_helper import _lazy_re_compile
@BaseSpatialField.register_lookup
class SameAsLookup(GISLookup):
    """
    The "~=" operator is the "same as" operator. It tests actual geometric
    equality of two features. So if A and B are the same feature,
    vertex-by-vertex, the operator returns true.
    """
    lookup_name = 'same_as'