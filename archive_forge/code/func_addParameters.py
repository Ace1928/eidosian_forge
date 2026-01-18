from .parameterTypes import GroupParameterItem
from ..Qt import QtCore, QtWidgets, mkQApp
from ..widgets.TreeWidget import TreeWidget
from .ParameterItem import ParameterItem
def addParameters(self, param, root=None, depth=0, showTop=True):
    """
        Adds one top-level :class:`Parameter <pyqtgraph.parametertree.Parameter>`
        to the view. 
        
        ============== ==========================================================
        **Arguments:** 
        param          The :class:`Parameter <pyqtgraph.parametertree.Parameter>` 
                       to add.
        root           The item within the tree to which *param* should be added.
                       By default, *param* is added as a top-level item.
        showTop        If False, then *param* will be hidden, and only its 
                       children will be visible in the tree.
        ============== ==========================================================
        """
    item = param.makeTreeItem(depth=depth)
    if root is None:
        root = self.invisibleRootItem()
        if not showTop:
            item.setText(0, '')
            item.setSizeHint(0, QtCore.QSize(1, 1))
            item.setSizeHint(1, QtCore.QSize(1, 1))
            depth -= 1
    root.addChild(item)
    item.treeWidgetChanged()
    for ch in param:
        self.addParameters(ch, root=item, depth=depth + 1)