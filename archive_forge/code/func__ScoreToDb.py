from rdkit.DataStructs.BitEnsemble import BitEnsemble
def _ScoreToDb(self, sig, dbConn, tableName=None, id=None, act=None):
    """ scores the "signature" that is passed in and puts the
  results in the db table

  """
    if tableName is None:
        try:
            tableName = self._dbTableName
        except AttributeError:
            raise ValueError('table name not set in BitEnsemble pre call to ScoreToDb()')
    if id is not None:
        cols = [id]
    else:
        cols = []
    score = 0
    for bit in self.GetBits():
        b = sig[bit]
        cols.append(b)
        score += b
    if act is not None:
        cols.append(act)
    dbConn.InsertData(tableName, cols)