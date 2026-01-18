from ..Qt import QtCore, QtWidgets
def addTopLevelItems(self, items):
    QtWidgets.QTreeWidget.addTopLevelItems(self, items)
    for item in items:
        self.informTreeWidgetChange(item)