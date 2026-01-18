from rdkit.Dbase import DbInfo, DbModule, DbUtils
def InsertColumnData(self, tableName, columnName, value, where):
    """ inserts data into a particular column of the table

    **Arguments**

      - tableName: the name of the table to manipulate

      - columnName: name of the column to update

      - value: the value to insert

      - where: a query yielding the row where the data should be inserted

    """
    c = self.GetCursor()
    cmd = 'update %s set %s=%s where %s' % (tableName, columnName, DbModule.placeHolder, where)
    c.execute(cmd, (value,))