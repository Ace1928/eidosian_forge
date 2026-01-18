import sys
from io import StringIO
from rdkit.Dbase import DbInfo, DbModule
from rdkit.Dbase.DbResultSet import DbResultSet, RandomAccessDbResultSet
def _insertBlock(conn, sqlStr, block, silent=False):
    try:
        conn.cursor().executemany(sqlStr, block)
    except Exception:
        res = 0
        conn.commit()
        for row in block:
            try:
                conn.cursor().execute(sqlStr, tuple(row))
                res += 1
            except Exception:
                if not silent:
                    import traceback
                    traceback.print_exc()
                    print('insert failed:', sqlStr)
                    print('\t', repr(row))
            else:
                conn.commit()
    else:
        res = len(block)
    return res