from django.db.models import Transform
from django.db.models.lookups import PostgresOperatorLookup
from django.db.models.sql.query import Query
from .search import SearchVector, SearchVectorExact, SearchVectorField
class ContainedBy(PostgresOperatorLookup):
    lookup_name = 'contained_by'
    postgres_operator = '<@'