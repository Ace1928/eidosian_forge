from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
class GeometryField(BaseSpatialField):
    """
    The base Geometry field -- maps to the OpenGIS Specification Geometry type.
    """
    description = _('The base Geometry field — maps to the OpenGIS Specification Geometry type.')
    form_class = forms.GeometryField
    geom_type = 'GEOMETRY'
    geom_class = None

    def __init__(self, verbose_name=None, dim=2, geography=False, *, extent=(-180.0, -90.0, 180.0, 90.0), tolerance=0.05, **kwargs):
        """
        The initialization function for geometry fields. In addition to the
        parameters from BaseSpatialField, it takes the following as keyword
        arguments:

        dim:
         The number of dimensions for this geometry.  Defaults to 2.

        extent:
         Customize the extent, in a 4-tuple of WGS 84 coordinates, for the
         geometry field entry in the `USER_SDO_GEOM_METADATA` table.  Defaults
         to (-180.0, -90.0, 180.0, 90.0).

        tolerance:
         Define the tolerance, in meters, to use for the geometry field
         entry in the `USER_SDO_GEOM_METADATA` table.  Defaults to 0.05.
        """
        self.dim = dim
        self.geography = geography
        self._extent = extent
        self._tolerance = tolerance
        super().__init__(verbose_name=verbose_name, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.dim != 2:
            kwargs['dim'] = self.dim
        if self.geography is not False:
            kwargs['geography'] = self.geography
        if self._extent != (-180.0, -90.0, 180.0, 90.0):
            kwargs['extent'] = self._extent
        if self._tolerance != 0.05:
            kwargs['tolerance'] = self._tolerance
        return (name, path, args, kwargs)

    def contribute_to_class(self, cls, name, **kwargs):
        super().contribute_to_class(cls, name, **kwargs)
        setattr(cls, self.attname, SpatialProxy(self.geom_class or GEOSGeometry, self, load_func=GEOSGeometry))

    def formfield(self, **kwargs):
        defaults = {'form_class': self.form_class, 'geom_type': self.geom_type, 'srid': self.srid, **kwargs}
        if self.dim > 2 and (not getattr(defaults['form_class'].widget, 'supports_3d', False)):
            defaults.setdefault('widget', forms.Textarea)
        return super().formfield(**defaults)

    def select_format(self, compiler, sql, params):
        """
        Return the selection format string, depending on the requirements
        of the spatial backend. For example, Oracle and MySQL require custom
        selection formats in order to retrieve geometries in OGC WKB.
        """
        if not compiler.query.subquery:
            return (compiler.connection.ops.select % sql, params)
        return (sql, params)