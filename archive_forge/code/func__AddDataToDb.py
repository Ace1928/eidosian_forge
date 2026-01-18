import sys
from io import StringIO
from rdkit.Dbase import DbInfo, DbModule
from rdkit.Dbase.DbResultSet import DbResultSet, RandomAccessDbResultSet
def _AddDataToDb(dBase, table, user, password, colDefs, colTypes, data, nullMarker=None, blockSize=100, cn=None):
    """ *For Internal Use*

      (drops and) creates a table and then inserts the values

    """
    if not cn:
        cn = DbModule.connect(dBase, user, password)
    c = cn.cursor()
    try:
        c.execute('drop table %s' % table)
    except Exception:
        print('cannot drop table %s' % table)
    try:
        sqlStr = 'create table %s (%s)' % (table, colDefs)
        c.execute(sqlStr)
    except Exception:
        print('create table failed: ', sqlStr)
        print('here is the exception:')
        import traceback
        traceback.print_exc()
        return
    cn.commit()
    c = None
    block = []
    entryTxt = [DbModule.placeHolder] * len(data[0])
    dStr = ','.join(entryTxt)
    sqlStr = 'insert into %s values (%s)' % (table, dStr)
    nDone = 0
    for row in data:
        entries = [None] * len(row)
        for col in range(len(row)):
            if row[col] is not None and (nullMarker is None or row[col] != nullMarker):
                if colTypes[col][0] == float:
                    entries[col] = float(row[col])
                elif colTypes[col][0] == int:
                    entries[col] = int(row[col])
                else:
                    entries[col] = str(row[col])
            else:
                entries[col] = None
        block.append(tuple(entries))
        if len(block) >= blockSize:
            nDone += _insertBlock(cn, sqlStr, block)
            if not hasattr(cn, 'autocommit') or cn.autocommit != True:
                cn.commit()
            block = []
    if len(block):
        nDone += _insertBlock(cn, sqlStr, block)
    if not hasattr(cn, 'autocommit') or cn.autocommit != True:
        cn.commit()