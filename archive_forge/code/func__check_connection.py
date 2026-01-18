from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
def _check_connection(self, connection):
    if not connection.features.gis_enabled or not connection.features.supports_raster:
        raise ImproperlyConfigured('Raster fields require backends with raster support.')