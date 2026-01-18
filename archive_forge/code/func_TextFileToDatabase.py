import sys
from io import StringIO
from rdkit.Dbase import DbInfo, DbModule
from rdkit.Dbase.DbResultSet import DbResultSet, RandomAccessDbResultSet
def TextFileToDatabase(dBase, table, inF, delim=',', user='sysdba', password='masterkey', maxColLabelLen=31, keyCol=None, nullMarker=None):
    """loads the contents of the text file into a database.

      **Arguments**

        - dBase: the name of the DB to use

        - table: the name of the table to create/overwrite

        - inF: the file like object from which the data should
          be pulled (must support readline())

        - delim: the delimiter used to separate fields

        - user: the user name to use in connecting to the DB

        - password: the password to use in connecting to the DB

        - maxColLabelLen: the maximum length a column label should be
          allowed to have (truncation otherwise)

        - keyCol: the column to be used as an index for the db

      **Notes**

        - if _table_ already exists, it is destroyed before we write
          the new data

        - we assume that the first row of the file contains the column names

    """
    table.replace('-', '_')
    table.replace(' ', '_')
    colHeadings = inF.readline().split(delim)
    _AdjustColHeadings(colHeadings, maxColLabelLen)
    nCols = len(colHeadings)
    data = []
    inL = inF.readline()
    while inL:
        inL = inL.replace('\r', '')
        inL = inL.replace('\n', '')
        splitL = inL.split(delim)
        if len(splitL) != nCols:
            print('>>>', repr(inL))
            assert len(splitL) == nCols, 'unequal length'
        tmpVect = []
        for entry in splitL:
            try:
                val = int(entry)
            except ValueError:
                try:
                    val = float(entry)
                except ValueError:
                    val = entry
            tmpVect.append(val)
        data.append(tmpVect)
        inL = inF.readline()
    nRows = len(data)
    colTypes = TypeFinder(data, nRows, nCols, nullMarker=nullMarker)
    typeStrs = GetTypeStrings(colHeadings, colTypes, keyCol=keyCol)
    colDefs = ','.join(typeStrs)
    _AddDataToDb(dBase, table, user, password, colDefs, colTypes, data, nullMarker=nullMarker)