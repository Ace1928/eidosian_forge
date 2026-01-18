from django.contrib.gis.db.models.fields import BaseSpatialField
from django.contrib.gis.measure import Distance
from django.db import NotSupportedError
from django.db.models import Expression, Lookup, Transform
from django.db.models.sql.query import Query
from django.utils.regex_helper import _lazy_re_compile
@BaseSpatialField.register_lookup
class DWithinLookup(DistanceLookupBase):
    lookup_name = 'dwithin'
    sql_template = '%(func)s(%(lhs)s, %(rhs)s, %(value)s)'

    def process_distance(self, compiler, connection):
        dist_param = self.rhs_params[0]
        if not connection.features.supports_dwithin_distance_expr and hasattr(dist_param, 'resolve_expression') and (not isinstance(dist_param, Distance)):
            raise NotSupportedError('This backend does not support expressions for specifying distance in the dwithin lookup.')
        return super().process_distance(compiler, connection)

    def process_rhs(self, compiler, connection):
        dist_sql, dist_params = self.process_distance(compiler, connection)
        self.template_params['value'] = dist_sql
        rhs_sql, params = super().process_rhs(compiler, connection)
        return (rhs_sql, params + dist_params)