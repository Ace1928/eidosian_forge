from warnings import warn
from rdkit import RDConfig
from rdkit.Dbase import DbModule
from rdkit.Dbase.DbConnection import DbConnect
def GetDataSetInfo(self, **kwargs):
    """ Returns a MLDataSet pulled from a database using our stored
    values.

    """
    conn = DbConnect(self.dbName, self.tableName)
    res = conn.GetColumnNamesAndTypes(join=self.dbJoin, what=self.dbWhat, where=self.dbWhere)
    return res