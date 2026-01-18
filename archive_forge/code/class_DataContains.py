from django.db.models import Transform
from django.db.models.lookups import PostgresOperatorLookup
from django.db.models.sql.query import Query
from .search import SearchVector, SearchVectorExact, SearchVectorField
class DataContains(PostgresOperatorLookup):
    lookup_name = 'contains'
    postgres_operator = '@>'