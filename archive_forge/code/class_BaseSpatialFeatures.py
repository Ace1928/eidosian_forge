import re
from django.contrib.gis.db import models
class BaseSpatialFeatures:
    gis_enabled = True
    has_spatialrefsys_table = True
    supports_add_srs_entry = True
    supports_geometry_field_introspection = True
    supports_geography = False
    supports_3d_storage = False
    supports_3d_functions = False
    supports_transform = True
    supports_null_geometries = True
    supports_empty_geometries = False
    supports_distance_geodetic = True
    supports_length_geodetic = True
    supports_perimeter_geodetic = False
    supports_area_geodetic = True
    supports_num_points_poly = True
    supports_dwithin_distance_expr = True
    supports_raster = False
    supports_geometry_field_unique_index = True
    can_alter_geometry_field = True
    supports_tolerance_parameter = False
    unsupported_geojson_options = {}
    empty_intersection_returns_none = True

    @property
    def supports_bbcontains_lookup(self):
        return 'bbcontains' in self.connection.ops.gis_operators

    @property
    def supports_contained_lookup(self):
        return 'contained' in self.connection.ops.gis_operators

    @property
    def supports_crosses_lookup(self):
        return 'crosses' in self.connection.ops.gis_operators

    @property
    def supports_distances_lookups(self):
        return self.has_Distance_function

    @property
    def supports_dwithin_lookup(self):
        return 'dwithin' in self.connection.ops.gis_operators

    @property
    def supports_relate_lookup(self):
        return 'relate' in self.connection.ops.gis_operators

    @property
    def supports_isvalid_lookup(self):
        return self.has_IsValid_function

    @property
    def supports_collect_aggr(self):
        return models.Collect not in self.connection.ops.disallowed_aggregates

    @property
    def supports_extent_aggr(self):
        return models.Extent not in self.connection.ops.disallowed_aggregates

    @property
    def supports_make_line_aggr(self):
        return models.MakeLine not in self.connection.ops.disallowed_aggregates

    @property
    def supports_union_aggr(self):
        return models.Union not in self.connection.ops.disallowed_aggregates

    def __getattr__(self, name):
        m = re.match('has_(\\w*)_function$', name)
        if m:
            func_name = m[1]
            return func_name not in self.connection.ops.unsupported_functions
        raise AttributeError