from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
class AggregateQuery(Query):
    """
    Take another query as a parameter to the FROM clause and only select the
    elements in the provided list.
    """
    compiler = 'SQLAggregateCompiler'

    def __init__(self, model, inner_query):
        self.inner_query = inner_query
        super().__init__(model)