import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_commented_qresult(self):
    """Yield ``QueryResult`` objects from a commented file (PRIVATE)."""
    while True:
        comments = self._parse_comments()
        if comments:
            try:
                self.fields = comments['fields']
                qres_iter = self._parse_qresult()
            except KeyError:
                assert 'fields' not in comments
                qres_iter = iter([QueryResult()])
            for qresult in qres_iter:
                for key, value in comments.items():
                    setattr(qresult, key, value)
                yield qresult
        else:
            break