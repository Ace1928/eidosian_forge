from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _flatten_up_to_token(self, token):
    """Yields all tokens up to token but excluding current."""
    if token.is_group:
        token = next(token.flatten())
    for t in self._curr_stmt.flatten():
        if t == token:
            break
        yield t