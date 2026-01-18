import sys
from rdkit.Dbase import DbInfo
def _initColumnNamesAndTypes(self):
    self.colNames = []
    self.colTypes = []
    for cName, cType in DbInfo.GetColumnInfoFromCursor(self.cursor):
        self.colNames.append(cName)
        self.colTypes.append(cType)
    self.colNames = tuple(self.colNames)
    self.colTypes = tuple(self.colTypes)