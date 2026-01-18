from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
class BaseSpatialField(Field):
    """
    The Base GIS Field.

    It's used as a base class for GeometryField and RasterField. Defines
    properties that are common to all GIS fields such as the characteristics
    of the spatial reference system of the field.
    """
    description = _('The base GIS field.')
    empty_strings_allowed = False

    def __init__(self, verbose_name=None, srid=4326, spatial_index=True, **kwargs):
        """
        The initialization function for base spatial fields. Takes the following
        as keyword arguments:

        srid:
         The spatial reference system identifier, an OGC standard.
         Defaults to 4326 (WGS84).

        spatial_index:
         Indicates whether to create a spatial index.  Defaults to True.
         Set this instead of 'db_index' for geographic fields since index
         creation is different for geometry columns.
        """
        self.spatial_index = spatial_index
        self.srid = srid
        kwargs['verbose_name'] = verbose_name
        super().__init__(**kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['srid'] = self.srid
        if self.spatial_index is not True:
            kwargs['spatial_index'] = self.spatial_index
        return (name, path, args, kwargs)

    def db_type(self, connection):
        return connection.ops.geo_db_type(self)

    def spheroid(self, connection):
        return get_srid_info(self.srid, connection).spheroid

    def units(self, connection):
        return get_srid_info(self.srid, connection).units

    def units_name(self, connection):
        return get_srid_info(self.srid, connection).units_name

    def geodetic(self, connection):
        """
        Return true if this field's SRID corresponds with a coordinate
        system that uses non-projected units (e.g., latitude/longitude).
        """
        return get_srid_info(self.srid, connection).geodetic

    def get_placeholder(self, value, compiler, connection):
        """
        Return the placeholder for the spatial column for the
        given value.
        """
        return connection.ops.get_geom_placeholder(self, value, compiler)

    def get_srid(self, obj):
        """
        Return the default SRID for the given geometry or raster, taking into
        account the SRID set for the field. For example, if the input geometry
        or raster doesn't have an SRID, then the SRID of the field will be
        returned.
        """
        srid = obj.srid
        if srid is None or self.srid == -1 or (srid == -1 and self.srid != -1):
            return self.srid
        else:
            return srid

    def get_db_prep_value(self, value, connection, *args, **kwargs):
        if value is None:
            return None
        return connection.ops.Adapter(super().get_db_prep_value(value, connection, *args, **kwargs), **{'geography': True} if self.geography and connection.features.supports_geography else {})

    def get_raster_prep_value(self, value, is_candidate):
        """
        Return a GDALRaster if conversion is successful, otherwise return None.
        """
        if isinstance(value, gdal.GDALRaster):
            return value
        elif is_candidate:
            try:
                return gdal.GDALRaster(value)
            except GDALException:
                pass
        elif isinstance(value, dict):
            try:
                return gdal.GDALRaster(value)
            except GDALException:
                raise ValueError("Couldn't create spatial object from lookup value '%s'." % value)

    def get_prep_value(self, value):
        obj = super().get_prep_value(value)
        if obj is None:
            return None
        if isinstance(obj, GEOSGeometry):
            pass
        else:
            is_candidate = isinstance(obj, (bytes, str)) or hasattr(obj, '__geo_interface__')
            raster = self.get_raster_prep_value(obj, is_candidate)
            if raster:
                obj = raster
            elif is_candidate:
                try:
                    obj = GEOSGeometry(obj)
                except (GEOSException, GDALException):
                    raise ValueError("Couldn't create spatial object from lookup value '%s'." % obj)
            else:
                raise ValueError('Cannot use object with type %s for a spatial lookup parameter.' % type(obj).__name__)
        obj.srid = self.get_srid(obj)
        return obj