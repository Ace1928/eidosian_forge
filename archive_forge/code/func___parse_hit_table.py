import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def __parse_hit_table(self):
    """Parse hit table rows."""
    hit_rows = []
    while True:
        line = self.handle.readline()
        if not line or line.strip():
            break
        hit_rows.append('')
    self.line = line
    return hit_rows