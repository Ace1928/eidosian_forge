import sys
from rdkit.Dbase import DbInfo
class DbResultBase(object):

    def __init__(self, cursor, conn, cmd, removeDups=-1, transform=None, extras=None):
        self.cursor = cursor
        self.removeDups = removeDups
        self.transform = transform
        self.cmd = cmd
        self.conn = conn
        self.extras = extras
        self.Reset()
        self._initColumnNamesAndTypes()
        self.Reset()

    def Reset(self):
        """ implement in subclasses

    """
        try:
            if not self.extras:
                self.cursor.execute(self.cmd)
            else:
                self.cursor.execute(self.cmd, self.extras)
        except Exception:
            sys.stderr.write('the command "%s" generated errors:\n' % self.cmd)
            import traceback
            traceback.print_exc()

    def __iter__(self):
        self.Reset()
        return self

    def _initColumnNamesAndTypes(self):
        self.colNames = []
        self.colTypes = []
        for cName, cType in DbInfo.GetColumnInfoFromCursor(self.cursor):
            self.colNames.append(cName)
            self.colTypes.append(cType)
        self.colNames = tuple(self.colNames)
        self.colTypes = tuple(self.colTypes)

    def GetColumnNames(self):
        return self.colNames

    def GetColumnTypes(self):
        return self.colTypes

    def GetColumnNamesAndTypes(self):
        return tuple((nt for nt in zip(self.colNames, self.colTypes)))