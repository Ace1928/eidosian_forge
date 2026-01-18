from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
class OutputFilter(object):
    varname_prefix = ''

    def __init__(self, varname='sql'):
        self.varname = self.varname_prefix + varname
        self.count = 0

    def _process(self, stream, varname, has_nl):
        raise NotImplementedError

    def process(self, stmt):
        self.count += 1
        if self.count > 1:
            varname = u'{f.varname}{f.count}'.format(f=self)
        else:
            varname = self.varname
        has_nl = len(text_type(stmt).strip().splitlines()) > 1
        stmt.tokens = self._process(stmt.tokens, varname, has_nl)
        return stmt