import sys
from rdkit import Chem
from rdkit.Chem.Suppliers.MolSupplier import MolSupplier
class DbMolSupplier(MolSupplier):
    """
    new molecules come back with all additional fields from the
    database set in a "_fieldsFromDb" data member

  """

    def __init__(self, dbResults, molColumnFormats={'SMILES': 'SMI', 'SMI': 'SMI', 'MOLPKL': 'PKL'}, nameCol='', transformFunc=None, **kwargs):
        """

      DbResults should be a subclass of Dbase.DbResultSet.DbResultBase

    """
        self._data = dbResults
        self._colNames = [x.upper() for x in self._data.GetColumnNames()]
        nameCol = nameCol.upper()
        self.molCol = -1
        self.transformFunc = transformFunc
        try:
            self.nameCol = self._colNames.index(nameCol)
        except ValueError:
            self.nameCol = -1
        for name in molColumnFormats:
            name = name.upper()
            try:
                idx = self._colNames.index(name)
            except ValueError:
                pass
            else:
                self.molCol = idx
                self.molFmt = molColumnFormats[name]
                break
        if self.molCol < 0:
            raise ValueError('DbResultSet has no recognizable molecule column')
        del self._colNames[self.molCol]
        self._colNames = tuple(self._colNames)
        self._numProcessed = 0

    def GetColumnNames(self):
        return self._colNames

    def _BuildMol(self, data):
        data = list(data)
        molD = data[self.molCol]
        del data[self.molCol]
        self._numProcessed += 1
        try:
            if self.molFmt == 'SMI':
                newM = Chem.MolFromSmiles(str(molD))
                if not newM:
                    warning('Problems processing mol %d, smiles: %s\n' % (self._numProcessed, molD))
            elif self.molFmt == 'PKL':
                newM = Chem.Mol(str(molD))
        except Exception:
            import traceback
            traceback.print_exc()
            newM = None
        else:
            if newM and self.transformFunc:
                try:
                    newM = self.transformFunc(newM, data)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    newM = None
            if newM:
                newM._fieldsFromDb = data
                nFields = len(data)
                for i in range(nFields):
                    newM.SetProp(self._colNames[i], str(data[i]))
                if self.nameCol >= 0:
                    newM.SetProp('_Name', str(data[self.nameCol]))
        return newM