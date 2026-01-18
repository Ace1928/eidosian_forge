from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.GraphicsView import GraphicsView
class FlowchartViewBox(ViewBox):

    def __init__(self, widget, *args, **kwargs):
        ViewBox.__init__(self, *args, **kwargs)
        self.widget = widget

    def getMenu(self, ev):
        self._fc_menu = QtWidgets.QMenu()
        self._subMenus = self.getContextMenus(ev)
        for menu in self._subMenus:
            self._fc_menu.addMenu(menu)
        return self._fc_menu

    def getContextMenus(self, ev):
        menu = self.widget.buildMenu(ev.scenePos())
        menu.setTitle(translate('Context Menu', 'Add node'))
        return [menu, ViewBox.getMenu(self, ev)]