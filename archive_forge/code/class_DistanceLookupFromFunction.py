from django.contrib.gis.db.models.fields import BaseSpatialField
from django.contrib.gis.measure import Distance
from django.db import NotSupportedError
from django.db.models import Expression, Lookup, Transform
from django.db.models.sql.query import Query
from django.utils.regex_helper import _lazy_re_compile
class DistanceLookupFromFunction(DistanceLookupBase):

    def as_sql(self, compiler, connection):
        spheroid = len(self.rhs_params) == 2 and self.rhs_params[-1] == 'spheroid' or None
        distance_expr = connection.ops.distance_expr_for_lookup(self.lhs, self.rhs, spheroid=spheroid)
        sql, params = compiler.compile(distance_expr.resolve_expression(compiler.query))
        dist_sql, dist_params = self.process_distance(compiler, connection)
        return ('%(func)s %(op)s %(dist)s' % {'func': sql, 'op': self.op, 'dist': dist_sql}, params + dist_params)