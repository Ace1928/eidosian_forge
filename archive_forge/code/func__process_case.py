from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _process_case(self, tlist):
    iterable = iter(tlist.get_cases())
    cond, _ = next(iterable)
    first = next(cond[0].flatten())
    with offset(self, self._get_offset(tlist[0])):
        with offset(self, self._get_offset(first)):
            for cond, value in iterable:
                token = value[0] if cond is None else cond[0]
                tlist.insert_before(token, self.nl())
            with offset(self, len('WHEN ')):
                self._process_default(tlist)
        end_idx, end = tlist.token_next_by(m=sql.Case.M_CLOSE)
        if end_idx is not None:
            tlist.insert_before(end_idx, self.nl())