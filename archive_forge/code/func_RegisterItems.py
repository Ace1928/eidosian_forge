from rdkit import RDConfig
from rdkit.Dbase import DbModule
def RegisterItems(conn, table, values, columnName, rows, startId='', idColName='Id', leadText='RDCmpd'):
    """
  """
    if rows and len(rows) != len(values):
        raise ValueError('length mismatch between rows and values')
    nVals = len(values)
    origOrder = {}
    for i, v in enumerate(values):
        origOrder[v] = i
    curs = conn.GetCursor()
    qs = ','.join(DbModule.placeHolder * nVals)
    curs.execute('create temporary table regitemstemp (%(columnName)s)' % locals())
    curs.executemany('insert into regitemstemp values (?)', [(x,) for x in values])
    query = 'select %(columnName)s,%(idColName)s from %(table)s ' + 'where %(columnName)s in (select * from regitemstemp)' % locals()
    curs.execute(query)
    dbData = curs.fetchall()
    if dbData and len(dbData) == nVals:
        return (0, [x[1] for x in dbData])
    if not startId:
        startId = GetNextRDId(conn, table, idColName=idColName, leadText=leadText)
        startId = RDIdToInt(startId)
    ids = [None] * nVals
    for val, ID in dbData:
        ids[origOrder[val]] = ID
    rowsToInsert = []
    for i in range(nVals):
        if ids[i] is None:
            ID = startId
            startId += 1
            ID = IndexToRDId(ID, leadText=leadText)
            ids[i] = ID
            if rows:
                row = [ID]
                row.extend(rows[i])
                rowsToInsert.append(row)
    if rowsToInsert:
        nCols = len(rowsToInsert[0])
        qs = ','.join(DbModule.placeHolder * nCols)
        curs.executemany('insert into %(table)s values (%(qs)s)' % locals(), rowsToInsert)
        conn.Commit()
    return (len(values) - len(dbData), ids)