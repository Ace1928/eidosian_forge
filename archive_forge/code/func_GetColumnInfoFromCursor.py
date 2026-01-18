import sys
from rdkit import RDConfig
from rdkit.Dbase import DbModule
def GetColumnInfoFromCursor(cursor):
    if cursor is None or cursor.description is None:
        return []
    results = []
    if not RDConfig.useSqlLite:
        for item in cursor.description:
            cName = item[0]
            cType = item[1]
            if cType in sqlTextTypes:
                typeStr = 'string'
            elif cType in sqlIntTypes:
                typeStr = 'integer'
            elif cType in sqlFloatTypes:
                typeStr = 'float'
            elif cType in sqlBinTypes:
                typeStr = 'binary'
            else:
                sys.stderr.write('odd type in col %s: %s\n' % (cName, str(cType)))
            results.append((cName, typeStr))
    else:
        r = cursor.fetchone()
        if not r:
            return results
        for i, v in enumerate(r):
            cName = cursor.description[i][0]
            typ = type(v)
            if isinstance(v, str):
                typeStr = 'string'
            elif typ == int:
                typeStr = 'integer'
            elif typ == float:
                typeStr = 'float'
            elif typ in (memoryview, bytes):
                typeStr = 'binary'
            else:
                sys.stderr.write('odd type in col %s: %s\n' % (cName, typ))
            results.append((cName, typeStr))
    return results