from rdkit import RDConfig
from rdkit.Dbase import DbModule
def GetNextId(conn, table, idColName='Id'):
    """ returns the next available Id in the database

  see RegisterItem for testing/documentation

  """
    vals = conn.GetData(table=table, fields=idColName)
    maxVal = 0
    for val in vals:
        val = RDIdToInt(val[0], validate=0)
        if val > maxVal:
            maxVal = val
    maxVal += 1
    return maxVal