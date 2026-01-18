from rdkit.DataStructs.BitEnsemble import BitEnsemble
def _InitScoreTable(self, dbConn, tableName, idInfo='', actInfo=''):
    """ inializes a db table to store our scores

    idInfo and actInfo should be strings with the definitions of the id and
    activity columns of the table (when desired)

  """
    if idInfo:
        cols = [idInfo]
    else:
        cols = []
    for bit in self.GetBits():
        cols.append('Bit_%d smallint' % bit)
    if actInfo:
        cols.append(actInfo)
    dbConn.AddTable(tableName, ','.join(cols))
    self._dbTableName = tableName