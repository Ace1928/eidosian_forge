from django.contrib.gis.db.models.fields import BaseSpatialField
from django.contrib.gis.measure import Distance
from django.db import NotSupportedError
from django.db.models import Expression, Lookup, Transform
from django.db.models.sql.query import Query
from django.utils.regex_helper import _lazy_re_compile
@BaseSpatialField.register_lookup
class LeftLookup(GISLookup):
    """
    The 'left' operator returns true if A's bounding box is strictly to the left
    of B's bounding box.
    """
    lookup_name = 'left'