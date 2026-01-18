from django.db.models import (
from django.db.models.expressions import CombinedExpression, register_combinable_fields
from django.db.models.functions import Cast, Coalesce
class SearchQueryCombinable:
    BITAND = '&&'
    BITOR = '||'

    def _combine(self, other, connector, reversed):
        if not isinstance(other, SearchQueryCombinable):
            raise TypeError('SearchQuery can only be combined with other SearchQuery instances, got %s.' % type(other).__name__)
        if reversed:
            return CombinedSearchQuery(other, connector, self, self.config)
        return CombinedSearchQuery(self, connector, other, self.config)

    def __or__(self, other):
        return self._combine(other, self.BITOR, False)

    def __ror__(self, other):
        return self._combine(other, self.BITOR, True)

    def __and__(self, other):
        return self._combine(other, self.BITAND, False)

    def __rand__(self, other):
        return self._combine(other, self.BITAND, True)