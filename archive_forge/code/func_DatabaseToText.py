import sys
from io import StringIO
from rdkit.Dbase import DbInfo, DbModule
from rdkit.Dbase.DbResultSet import DbResultSet, RandomAccessDbResultSet
def DatabaseToText(dBase, table, fields='*', join='', where='', user='sysdba', password='masterkey', delim=',', cn=None):
    """ Pulls the contents of a database and makes a deliminted text file from them

      **Arguments**
        - dBase: the name of the DB file to be used

        - table: the name of the table to query

        - fields: the fields to select with the SQL query

        - join: the join clause of the SQL query
          (e.g. 'join foo on foo.bar=base.bar')

        - where: the where clause of the SQL query
          (e.g. 'where foo = 2' or 'where bar > 17.6')

        - user: the username for DB access

        - password: the password to be used for DB access

      **Returns**

        - the CSV data (as text)

    """
    if len(where) and where.strip().find('where') == -1:
        where = 'where %s' % where
    if len(join) and join.strip().find('join') == -1:
        join = 'join %s' % join
    sqlCommand = 'select %s from %s %s %s' % (fields, table, join, where)
    if not cn:
        cn = DbModule.connect(dBase, user, password)
    c = cn.cursor()
    c.execute(sqlCommand)
    headers = []
    colsToTake = []
    for i in range(len(c.description)):
        item = c.description[i]
        if item[1] not in DbInfo.sqlBinTypes:
            colsToTake.append(i)
            headers.append(item[0])
    lines = []
    lines.append(delim.join(headers))
    results = c.fetchall()
    for res in results:
        d = _take(res, colsToTake)
        lines.append(delim.join([str(x) for x in d]))
    return '\n'.join(lines)