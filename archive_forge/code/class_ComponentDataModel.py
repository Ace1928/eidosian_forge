import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
class ComponentDataModel(myqt.QAbstractItemModel):
    """
    This is a data model to provide the tree structure and information
    to the tree viewer
    """

    def __init__(self, parent, ui_data, columns=['name', 'value'], components=(Var, BooleanVar), editable=[]):
        super().__init__(parent)
        self.column = columns
        self._col_editable = editable
        self.ui_data = ui_data
        self.components = components
        self.update_model()

    def update_model(self):
        self.rootItems = []
        self._create_tree(o=self.ui_data.model)

    def _update_tree(self, parent=None, o=None):
        """
        Check tree structure against the Pyomo model to add or delete
        components as needed. The arguments are to be used in the recursive
        function. Entering into this don't specify any args.
        """
        if o is None and len(self.rootItems) > 0:
            parent = self.rootItems[0]
            o = parent.data
            for no in o.component_objects(descend_into=False):
                self._update_tree(parent=parent, o=no)
            return
        elif o is None:
            return
        item = parent.ids.get(id(o), None)
        if item is not None:
            for i in item.children:
                try:
                    if i.data.parent_block() is None:
                        i.parent.children.remove(i)
                        del i.parent.ids[id(i.data)]
                        del i
                except AttributeError:
                    pass
        if isinstance(o, Block._ComponentDataClass):
            if item is None:
                item = self._add_item(parent=parent, o=o)
            for no in o.component_objects(descend_into=False):
                self._update_tree(parent=item, o=no)
        elif isinstance(o, Block):
            if item is None:
                item = self._add_item(parent=parent, o=o)
            if hasattr(o.index_set(), 'is_constructed') and o.index_set().is_constructed():
                for key in sorted(o.keys()):
                    self._update_tree(parent=item, o=o[key])
        elif isinstance(o, self.components):
            if item is None:
                item = self._add_item(parent=parent, o=o)
            if hasattr(o.index_set(), 'is_constructed') and o.index_set().is_constructed():
                for key in sorted(o.keys()):
                    if key == None:
                        break
                    item2 = item.ids.get(id(o[key]), None)
                    if item2 is None:
                        item2 = self._add_item(parent=item, o=o[key])
                    item2._visited = True
        return

    def _create_tree(self, parent=None, o=None):
        """
        This create a model tree structure to display in a tree view.
        Args:
            parent: a ComponentDataItem underwhich to create a TreeItem
            o: A Pyomo component to add to the tree
        """
        if isinstance(o, Block._ComponentDataClass):
            item = self._add_item(parent=parent, o=o)
            for no in o.component_objects(descend_into=False):
                self._create_tree(parent=item, o=no)
        elif isinstance(o, Block):
            item = self._add_item(parent=parent, o=o)
            if hasattr(o.index_set(), 'is_constructed') and o.index_set().is_constructed():
                for key in sorted(o.keys()):
                    self._create_tree(parent=item, o=o[key])
        elif isinstance(o, self.components):
            item = self._add_item(parent=parent, o=o)
            if hasattr(o.index_set(), 'is_constructed') and o.index_set().is_constructed():
                for key in sorted(o.keys()):
                    if key == None:
                        break
                    self._add_item(parent=item, o=o[key])

    def _add_item(self, parent, o):
        """
        Add a root item if parent is None, otherwise add a child
        """
        if parent is None:
            item = self._add_root_item(o)
        else:
            item = parent.add_child(o)
        return item

    def _add_root_item(self, o):
        """
        Add a root tree item
        """
        item = ComponentDataItem(None, o, ui_data=self.ui_data)
        self.rootItems.append(item)
        return item

    def parent(self, index):
        if not index.isValid():
            return myqt.QtCore.QModelIndex()
        item = index.internalPointer()
        if item.parent is None:
            return myqt.QtCore.QModelIndex()
        else:
            return self.createIndex(0, 0, item.parent)

    def index(self, row, column, parent=myqt.QtCore.QModelIndex()):
        if not parent.isValid():
            return self.createIndex(row, column, self.rootItems[row])
        parentItem = parent.internalPointer()
        return self.createIndex(row, column, parentItem.children[row])

    def columnCount(self, parent=myqt.QtCore.QModelIndex()):
        """
        Return the number of columns
        """
        return len(self.column)

    def rowCount(self, parent=myqt.QtCore.QModelIndex()):
        if not parent.isValid():
            return len(self.rootItems)
        return len(parent.internalPointer().children)

    def data(self, index=myqt.QtCore.QModelIndex(), role=myqt.Qt.ItemDataRole.DisplayRole):
        if role == myqt.Qt.ItemDataRole.DisplayRole or role == myqt.Qt.ItemDataRole.EditRole:
            a = self.column[index.column()]
            return index.internalPointer().get(a)
        elif role == myqt.Qt.ItemDataRole.ToolTipRole:
            if self.column[index.column()] == 'name':
                o = index.internalPointer()
                if isinstance(o.data, Constraint._ComponentDataClass):
                    return o.get('expr')
                else:
                    return o.get('doc')
        elif role == myqt.Qt.ItemDataRole.ForegroundRole:
            if isinstance(index.internalPointer().data, (Block, Block._ComponentDataClass)):
                return myqt.QColor(myqt.QtCore.Qt.black)
            else:
                return myqt.QColor(myqt.QtCore.Qt.blue)
        else:
            return

    def headerData(self, i, orientation, role=myqt.Qt.ItemDataRole.DisplayRole):
        """
        Return the column headings for the horizontal header and
        index numbers for the vertical header.
        """
        if orientation == myqt.Qt.Orientation.Horizontal and role == myqt.Qt.ItemDataRole.DisplayRole:
            return self.column[i]
        return None

    def flags(self, index=myqt.QtCore.QModelIndex()):
        if self.column[index.column()] in self._col_editable:
            return myqt.Qt.ItemFlag.ItemIsEnabled | myqt.Qt.ItemFlag.ItemIsSelectable | myqt.Qt.ItemFlag.ItemIsEditable
        else:
            return myqt.Qt.ItemFlag.ItemIsEnabled | myqt.Qt.ItemFlag.ItemIsSelectable