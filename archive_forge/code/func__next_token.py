from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _next_token(self, tlist, idx=-1):
    split_words = ('FROM', 'STRAIGHT_JOIN$', 'JOIN$', 'AND', 'OR', 'GROUP BY', 'ORDER BY', 'UNION', 'VALUES', 'SET', 'BETWEEN', 'EXCEPT', 'HAVING', 'LIMIT')
    m_split = (T.Keyword, split_words, True)
    tidx, token = tlist.token_next_by(m=m_split, idx=idx)
    if token and token.normalized == 'BETWEEN':
        tidx, token = self._next_token(tlist, tidx)
        if token and token.normalized == 'AND':
            tidx, token = self._next_token(tlist, tidx)
    return (tidx, token)