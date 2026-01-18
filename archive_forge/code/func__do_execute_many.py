import re
import warnings
from . import err
def _do_execute_many(self, prefix, values, postfix, args, max_stmt_length, encoding):
    conn = self._get_db()
    escape = self._escape_args
    if isinstance(prefix, str):
        prefix = prefix.encode(encoding)
    if isinstance(postfix, str):
        postfix = postfix.encode(encoding)
    sql = bytearray(prefix)
    args = iter(args)
    v = values % escape(next(args), conn)
    if isinstance(v, str):
        v = v.encode(encoding, 'surrogateescape')
    sql += v
    rows = 0
    for arg in args:
        v = values % escape(arg, conn)
        if isinstance(v, str):
            v = v.encode(encoding, 'surrogateescape')
        if len(sql) + len(v) + len(postfix) + 1 > max_stmt_length:
            rows += self.execute(sql + postfix)
            sql = bytearray(prefix)
        else:
            sql += b','
        sql += v
    rows += self.execute(sql + postfix)
    self.rowcount = rows
    return rows