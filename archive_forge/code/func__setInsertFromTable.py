from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
def _setInsertFromTable(self, value):
    """Switch the function used to insert an element from the
        normal one to the misnested table one and back again"""
    self._insertFromTable = value
    if value:
        self.insertElement = self.insertElementTable
    else:
        self.insertElement = self.insertElementNormal