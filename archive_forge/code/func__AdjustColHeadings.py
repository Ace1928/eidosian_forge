import sys
from io import StringIO
from rdkit.Dbase import DbInfo, DbModule
from rdkit.Dbase.DbResultSet import DbResultSet, RandomAccessDbResultSet
def _AdjustColHeadings(colHeadings, maxColLabelLen):
    """ *For Internal Use*

      removes illegal characters from column headings
      and truncates those which are too long.

    """
    for i in range(len(colHeadings)):
        colHeadings[i] = colHeadings[i].strip()
        colHeadings[i] = colHeadings[i].replace(' ', '_')
        colHeadings[i] = colHeadings[i].replace('-', '_')
        colHeadings[i] = colHeadings[i].replace('.', '_')
        if len(colHeadings[i]) > maxColLabelLen:
            newHead = colHeadings[i].replace('_', '')
            newHead = newHead[:maxColLabelLen]
            print('\tHeading %s too long, changed to %s' % (colHeadings[i], newHead))
            colHeadings[i] = newHead
    return colHeadings