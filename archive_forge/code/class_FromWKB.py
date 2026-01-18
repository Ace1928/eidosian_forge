from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
class FromWKB(GeoFunc):
    output_field = GeometryField(srid=0)
    arity = 1
    geom_param_pos = ()