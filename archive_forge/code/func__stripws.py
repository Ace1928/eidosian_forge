from sqlparse import sql, tokens as T
from sqlparse.utils import split_unquoted_newlines
def _stripws(self, tlist):
    func_name = '_stripws_{cls}'.format(cls=type(tlist).__name__)
    func = getattr(self, func_name.lower(), self._stripws_default)
    func(tlist)