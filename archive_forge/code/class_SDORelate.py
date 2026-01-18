import re
from django.contrib.gis.db import models
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.oracle.adapter import OracleSpatialAdapter
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.geos.geometry import GEOSGeometry, GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.db.backends.oracle.operations import DatabaseOperations
class SDORelate(SpatialOperator):
    sql_template = "SDO_RELATE(%(lhs)s, %(rhs)s, 'mask=%(mask)s') = 'TRUE'"

    def check_relate_argument(self, arg):
        masks = 'TOUCH|OVERLAPBDYDISJOINT|OVERLAPBDYINTERSECT|EQUAL|INSIDE|COVEREDBY|CONTAINS|COVERS|ANYINTERACT|ON'
        mask_regex = re.compile('^(%s)(\\+(%s))*$' % (masks, masks), re.I)
        if not isinstance(arg, str) or not mask_regex.match(arg):
            raise ValueError('Invalid SDO_RELATE mask: "%s"' % arg)

    def as_sql(self, connection, lookup, template_params, sql_params):
        template_params['mask'] = sql_params[-1]
        return super().as_sql(connection, lookup, template_params, sql_params[:-1])