from ..Qt import QtCore, QtWidgets
class InvisibleRootItem(object):
    """Wrapper around a TreeWidget's invisible root item that calls
    TreeWidget.informTreeWidgetChange when child items are added/removed.
    """

    def __init__(self, item):
        self._real_item = item

    def addChild(self, child):
        self._real_item.addChild(child)
        TreeWidget.informTreeWidgetChange(child)

    def addChildren(self, childs):
        self._real_item.addChildren(childs)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)

    def insertChild(self, index, child):
        self._real_item.insertChild(index, child)
        TreeWidget.informTreeWidgetChange(child)

    def insertChildren(self, index, childs):
        self._real_item.addChildren(index, childs)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)

    def removeChild(self, child):
        self._real_item.removeChild(child)
        TreeWidget.informTreeWidgetChange(child)

    def takeChild(self, index):
        child = self._real_item.takeChild(index)
        TreeWidget.informTreeWidgetChange(child)
        return child

    def takeChildren(self):
        childs = self._real_item.takeChildren()
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)
        return childs

    def __getattr__(self, attr):
        return getattr(self._real_item, attr)