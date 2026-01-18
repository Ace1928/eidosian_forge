from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _split_statements(self, tlist):
    ttypes = (T.Keyword.DML, T.Keyword.DDL)
    tidx, token = tlist.token_next_by(t=ttypes)
    while token:
        pidx, prev_ = tlist.token_prev(tidx, skip_ws=False)
        if prev_ and prev_.is_whitespace:
            del tlist.tokens[pidx]
            tidx -= 1
        if prev_:
            tlist.insert_before(tidx, self.nl())
            tidx += 1
        tidx, token = tlist.token_next_by(t=ttypes, idx=tidx)