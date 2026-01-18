from django.db.models import Transform
from django.db.models.lookups import PostgresOperatorLookup
from django.db.models.sql.query import Query
from .search import SearchVector, SearchVectorExact, SearchVectorField
class TrigramWordSimilar(PostgresOperatorLookup):
    lookup_name = 'trigram_word_similar'
    postgres_operator = '%%>'