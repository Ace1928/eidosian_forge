from django.db.models import Transform
from django.db.models.lookups import PostgresOperatorLookup
from django.db.models.sql.query import Query
from .search import SearchVector, SearchVectorExact, SearchVectorField
class HasAnyKeys(HasKeys):
    lookup_name = 'has_any_keys'
    postgres_operator = '?|'