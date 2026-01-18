from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _process_parenthesis(self, tlist):
    ttypes = (T.Keyword.DML, T.Keyword.DDL)
    _, is_dml_dll = tlist.token_next_by(t=ttypes)
    fidx, first = tlist.token_next_by(m=sql.Parenthesis.M_OPEN)
    with indent(self, 1 if is_dml_dll else 0):
        tlist.tokens.insert(0, self.nl()) if is_dml_dll else None
        with offset(self, self._get_offset(first) + 1):
            self._process_default(tlist, not is_dml_dll)