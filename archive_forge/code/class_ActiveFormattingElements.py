from __future__ import absolute_import, division, unicode_literals
from six import text_type
from ..constants import scopingElements, tableInsertModeElements, namespaces
class ActiveFormattingElements(list):

    def append(self, node):
        equalCount = 0
        if node != Marker:
            for element in self[::-1]:
                if element == Marker:
                    break
                if self.nodesEqual(element, node):
                    equalCount += 1
                if equalCount == 3:
                    self.remove(element)
                    break
        list.append(self, node)

    def nodesEqual(self, node1, node2):
        if not node1.nameTuple == node2.nameTuple:
            return False
        if not node1.attributes == node2.attributes:
            return False
        return True