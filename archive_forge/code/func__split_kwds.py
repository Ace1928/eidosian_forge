from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _split_kwds(self, tlist):
    tidx, token = self._next_token(tlist)
    while token:
        pidx, prev_ = tlist.token_prev(tidx, skip_ws=False)
        uprev = text_type(prev_)
        if prev_ and prev_.is_whitespace:
            del tlist.tokens[pidx]
            tidx -= 1
        if not (uprev.endswith('\n') or uprev.endswith('\r')):
            tlist.insert_before(tidx, self.nl())
            tidx += 1
        tidx, token = self._next_token(tlist, tidx)