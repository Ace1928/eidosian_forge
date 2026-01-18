from django.db.models import Transform
from django.db.models.lookups import PostgresOperatorLookup
from django.db.models.sql.query import Query
from .search import SearchVector, SearchVectorExact, SearchVectorField
class HasKey(PostgresOperatorLookup):
    lookup_name = 'has_key'
    postgres_operator = '?'
    prepare_rhs = False