from rdkit import RDConfig
from rdkit.Dbase import DbModule
def RegisterItem(conn, table, value, columnName, data=None, id='', idColName='Id', leadText='RDCmpd'):
    """
  >>> from rdkit.Dbase.DbConnection import DbConnect
  >>> conn = DbConnect(tempDbName)
  >>> tblName = 'StorageTest'
  >>> conn.AddTable(tblName,'id varchar(32) not null primary key,label varchar(40),val int')
  >>> RegisterItem(conn,tblName,'label1','label',['label1',1])==(1, 'RDCmpd-000-001-1')
  True
  >>> RegisterItem(conn,tblName,'label2','label',['label2',1])==(1, 'RDCmpd-000-002-2')
  True
  >>> RegisterItem(conn,tblName,'label1','label',['label1',1])==(0, 'RDCmpd-000-001-1')
  True
  >>> str(GetNextRDId(conn,tblName))
  'RDCmpd-000-003-3'
  >>> tuple(conn.GetData(table=tblName)[0])==('RDCmpd-000-001-1', 'label1', 1)
  True

  It's also possible to provide ids by hand:

  >>> RegisterItem(conn,tblName,'label10','label',['label10',1],
  ...              id='RDCmpd-000-010-1')==(1, 'RDCmpd-000-010-1')
  True
  >>> str(GetNextRDId(conn,tblName))
  'RDCmpd-000-011-2'

  """
    curs = conn.GetCursor()
    query = 'select %s from %s where %s=%s' % (idColName, table, columnName, DbModule.placeHolder)
    curs.execute(query, (value,))
    tmp = curs.fetchone()
    if tmp:
        return (0, tmp[0])
    ID = id or GetNextRDId(conn, table, idColName=idColName, leadText=leadText)
    if data:
        row = [ID]
        row.extend(data)
        conn.InsertData(table, row)
        conn.Commit()
    return (1, ID)