from ..Qt import QtCore, QtWidgets
class TreeWidgetItem(QtWidgets.QTreeWidgetItem):
    """
    TreeWidgetItem that keeps track of its own widgets and expansion state.
    
      * Widgets may be added to columns before the item is added to a tree.
      * Expanded state may be set before item is added to a tree.
      * Adds setCheked and isChecked methods.
      * Adds addChildren, insertChildren, and takeChildren methods.
    """

    def __init__(self, *args):
        QtWidgets.QTreeWidgetItem.__init__(self, *args)
        self._widgets = {}
        self._tree = None
        self._expanded = False

    def setChecked(self, column, checked):
        self.setCheckState(column, QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)

    def isChecked(self, col):
        return self.checkState(col) == QtCore.Qt.CheckState.Checked

    def setExpanded(self, exp):
        self._expanded = exp
        QtWidgets.QTreeWidgetItem.setExpanded(self, exp)

    def isExpanded(self):
        return self._expanded

    def setWidget(self, column, widget):
        if column in self._widgets:
            self.removeWidget(column)
        self._widgets[column] = widget
        tree = self.treeWidget()
        if tree is None:
            return
        else:
            tree.setItemWidget(self, column, widget)

    def removeWidget(self, column):
        del self._widgets[column]
        tree = self.treeWidget()
        if tree is None:
            return
        tree.removeItemWidget(self, column)

    def treeWidgetChanged(self):
        tree = self.treeWidget()
        if self._tree is tree:
            return
        self._tree = self.treeWidget()
        if tree is None:
            return
        for col, widget in self._widgets.items():
            tree.setItemWidget(self, col, widget)
        QtWidgets.QTreeWidgetItem.setExpanded(self, self._expanded)

    def childItems(self):
        return [self.child(i) for i in range(self.childCount())]

    def addChild(self, child):
        QtWidgets.QTreeWidgetItem.addChild(self, child)
        TreeWidget.informTreeWidgetChange(child)

    def addChildren(self, childs):
        QtWidgets.QTreeWidgetItem.addChildren(self, childs)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)

    def insertChild(self, index, child):
        QtWidgets.QTreeWidgetItem.insertChild(self, index, child)
        TreeWidget.informTreeWidgetChange(child)

    def insertChildren(self, index, childs):
        QtWidgets.QTreeWidgetItem.addChildren(self, index, childs)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)

    def removeChild(self, child):
        QtWidgets.QTreeWidgetItem.removeChild(self, child)
        TreeWidget.informTreeWidgetChange(child)

    def takeChild(self, index):
        child = QtWidgets.QTreeWidgetItem.takeChild(self, index)
        TreeWidget.informTreeWidgetChange(child)
        return child

    def takeChildren(self):
        childs = QtWidgets.QTreeWidgetItem.takeChildren(self)
        for child in childs:
            TreeWidget.informTreeWidgetChange(child)
        return childs

    def setData(self, column, role, value):
        checkstate = self.checkState(column)
        text = self.text(column)
        QtWidgets.QTreeWidgetItem.setData(self, column, role, value)
        treewidget = self.treeWidget()
        if treewidget is None:
            return
        if role == QtCore.Qt.ItemDataRole.CheckStateRole and checkstate != self.checkState(column):
            treewidget.sigItemCheckStateChanged.emit(self, column)
        elif role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole) and text != self.text(column):
            treewidget.sigItemTextChanged.emit(self, column)

    def itemClicked(self, col):
        """Called when this item is clicked on.
        
        Override this method to react to user clicks.
        """