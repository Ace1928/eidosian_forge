import sys
from rdkit import RDConfig
from rdkit.Dbase import DbModule
def GetColumnNamesAndTypes(dBase, table, user='sysdba', password='masterkey', join='', what='*', cn=None):
    """ gets a list of columns available in a DB table along with their types

      **Arguments**

        - dBase: the name of the DB file to be used

        - table: the name of the table to query

        - user: the username for DB access

        - password: the password to be used for DB access

        - join: an optional join clause (omit the verb 'join')

        - what: an optional clause indicating what to select

      **Returns**

        - a list of 2-tuples containing:

            1) column name

            2) column type

    """
    if not cn:
        cn = DbModule.connect(dBase, user, password)
    c = cn.cursor()
    cmd = 'select %s from %s' % (what, table)
    if join:
        cmd += ' join %s' % join
    c.execute(cmd)
    return GetColumnInfoFromCursor(c)