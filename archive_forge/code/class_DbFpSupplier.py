import pickle
from rdkit import DataStructs
from rdkit.VLib.Node import VLibNode
class DbFpSupplier(VLibNode):
    """
      new fps come back with all additional fields from the
      database set in a "_fieldsFromDb" data member

    """

    def __init__(self, dbResults, fpColName='AutoFragmentFp', usePickles=True):
        """

          DbResults should be a subclass of Dbase.DbResultSet.DbResultBase

        """
        VLibNode.__init__(self)
        self._usePickles = usePickles
        self._data = dbResults
        self._fpColName = fpColName.upper()
        self._colNames = [x.upper() for x in self._data.GetColumnNames()]
        if self._fpColName not in self._colNames:
            raise ValueError('fp column name "%s" not found in result set: %s' % (self._fpColName, str(self._colNames)))
        self.fpCol = self._colNames.index(self._fpColName)
        del self._colNames[self.fpCol]
        self._colNames = tuple(self._colNames)
        self._numProcessed = 0

    def GetColumnNames(self):
        return self._colNames

    def _BuildFp(self, data):
        data = list(data)
        pkl = bytes(data[self.fpCol], encoding='Latin1')
        del data[self.fpCol]
        self._numProcessed += 1
        try:
            if self._usePickles:
                newFp = pickle.loads(pkl, encoding='bytes')
            else:
                newFp = DataStructs.ExplicitBitVect(pkl)
        except Exception:
            import traceback
            traceback.print_exc()
            newFp = None
        if newFp:
            newFp._fieldsFromDb = data
        return newFp

    def next(self):
        itm = self.NextItem()
        if itm is None:
            raise StopIteration
        return itm
    __next__ = next