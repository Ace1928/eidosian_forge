from rdkit.Dbase import DbInfo, DbModule, DbUtils
def GetDataCount(self, table=None, where='', join='', **kwargs):
    """ returns a count of the number of results a query will return

      **Arguments**

       - table: (optional) the table to use

       - where: the SQL where clause to be used with the DB query

       - join: the SQL join clause to be used with the DB query


      **Returns**

          an int

      **Notes**

        - this uses _DbUtils.GetData_

    """
    table = table or self.tableName
    return DbUtils.GetData(self.dbName, table, fieldString='count(*)', whereString=where, cn=self.cn, user=self.user, password=self.password, join=join, forceList=0)[0][0]