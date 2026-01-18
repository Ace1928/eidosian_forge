from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _process_default(self, tlist, stmts=True):
    self._split_statements(tlist) if stmts else None
    self._split_kwds(tlist)
    for sgroup in tlist.get_sublists():
        self._process(sgroup)