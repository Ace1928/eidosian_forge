import json
from django.contrib.gis.gdal import CoordTransform, SpatialReference
from django.core.serializers.base import SerializerDoesNotExist
from django.core.serializers.json import Serializer as JSONSerializer
def _init_options(self):
    super()._init_options()
    self.geometry_field = self.json_kwargs.pop('geometry_field', None)
    self.id_field = self.json_kwargs.pop('id_field', None)
    self.srid = self.json_kwargs.pop('srid', 4326)
    if self.selected_fields is not None and self.geometry_field is not None and (self.geometry_field not in self.selected_fields):
        self.selected_fields = [*self.selected_fields, self.geometry_field]