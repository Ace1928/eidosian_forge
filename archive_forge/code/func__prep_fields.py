import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _prep_fields(self, fields):
    """Validate and format the given fields for use by the parser (PRIVATE)."""
    if isinstance(fields, str):
        fields = fields.strip().split(' ')
    if 'std' in fields:
        idx = fields.index('std')
        fields = fields[:idx] + _DEFAULT_FIELDS + fields[idx + 1:]
    if not set(fields).intersection(_MIN_QUERY_FIELDS) or not set(fields).intersection(_MIN_HIT_FIELDS):
        raise ValueError('Required query and/or hit ID field not found.')
    return fields