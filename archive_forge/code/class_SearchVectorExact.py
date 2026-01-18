from django.db.models import (
from django.db.models.expressions import CombinedExpression, register_combinable_fields
from django.db.models.functions import Cast, Coalesce
class SearchVectorExact(Lookup):
    lookup_name = 'exact'

    def process_rhs(self, qn, connection):
        if not isinstance(self.rhs, (SearchQuery, CombinedSearchQuery)):
            config = getattr(self.lhs, 'config', None)
            self.rhs = SearchQuery(self.rhs, config=config)
        rhs, rhs_params = super().process_rhs(qn, connection)
        return (rhs, rhs_params)

    def as_sql(self, qn, connection):
        lhs, lhs_params = self.process_lhs(qn, connection)
        rhs, rhs_params = self.process_rhs(qn, connection)
        params = lhs_params + rhs_params
        return ('%s @@ %s' % (lhs, rhs), params)