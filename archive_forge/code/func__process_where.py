from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _process_where(self, tlist):
    tidx, token = tlist.token_next_by(m=(T.Keyword, 'WHERE'))
    tlist.insert_before(tidx, self.nl())
    with indent(self):
        self._process_default(tlist)