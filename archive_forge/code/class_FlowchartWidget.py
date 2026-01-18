import importlib
import os
from collections import OrderedDict
from numpy import ndarray
from .. import DataTreeWidget, FileDialog
from .. import configfile as configfile
from .. import dockarea as dockarea
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtWidgets
from . import FlowchartCtrlTemplate_generic as FlowchartCtrlTemplate
from . import FlowchartGraphicsView
from .library import LIBRARY
from .Node import Node
from .Terminal import Terminal
class FlowchartWidget(dockarea.DockArea):
    """Includes the actual graphical flowchart and debugging interface"""

    def __init__(self, chart, ctrl):
        dockarea.DockArea.__init__(self)
        self.chart = chart
        self.ctrl = ctrl
        self.hoverItem = None
        self.view = FlowchartGraphicsView.FlowchartGraphicsView(self)
        self.viewDock = dockarea.Dock('view', size=(1000, 600))
        self.viewDock.addWidget(self.view)
        self.viewDock.hideTitleBar()
        self.addDock(self.viewDock)
        self.hoverText = QtWidgets.QTextEdit()
        self.hoverText.setReadOnly(True)
        self.hoverDock = dockarea.Dock('Hover Info', size=(1000, 20))
        self.hoverDock.addWidget(self.hoverText)
        self.addDock(self.hoverDock, 'bottom')
        self.selInfo = QtWidgets.QWidget()
        self.selInfoLayout = QtWidgets.QGridLayout()
        self.selInfo.setLayout(self.selInfoLayout)
        self.selDescLabel = QtWidgets.QLabel()
        self.selNameLabel = QtWidgets.QLabel()
        self.selDescLabel.setWordWrap(True)
        self.selectedTree = DataTreeWidget()
        self.selInfoLayout.addWidget(self.selDescLabel)
        self.selInfoLayout.addWidget(self.selectedTree)
        self.selDock = dockarea.Dock('Selected Node', size=(1000, 200))
        self.selDock.addWidget(self.selInfo)
        self.addDock(self.selDock, 'bottom')
        self._scene = self.view.scene()
        self._viewBox = self.view.viewBox()
        self.buildMenu()
        self._scene.selectionChanged.connect(self.selectionChanged)
        self._scene.sigMouseHover.connect(self.hoverOver)

    def reloadLibrary(self):
        self.nodeMenu.triggered.disconnect(self.nodeMenuTriggered)
        self.nodeMenu = None
        self.subMenus = []
        self.chart.library.reload()
        self.buildMenu()

    def buildMenu(self, pos=None):

        def buildSubMenu(node, rootMenu, subMenus, pos=None):
            for section, node in node.items():
                if isinstance(node, OrderedDict):
                    menu = QtWidgets.QMenu(section)
                    rootMenu.addMenu(menu)
                    buildSubMenu(node, menu, subMenus, pos=pos)
                    subMenus.append(menu)
                else:
                    act = rootMenu.addAction(section)
                    act.nodeType = section
                    act.pos = pos
        self.nodeMenu = QtWidgets.QMenu()
        self.subMenus = []
        buildSubMenu(self.chart.library.getNodeTree(), self.nodeMenu, self.subMenus, pos=pos)
        self.nodeMenu.triggered.connect(self.nodeMenuTriggered)
        return self.nodeMenu

    def menuPosChanged(self, pos):
        self.menuPos = pos

    def showViewMenu(self, ev):
        self.buildMenu(ev.scenePos())
        self.nodeMenu.popup(ev.screenPos())

    def scene(self):
        return self._scene

    def viewBox(self):
        return self._viewBox

    def nodeMenuTriggered(self, action):
        nodeType = action.nodeType
        if action.pos is not None:
            pos = action.pos
        else:
            pos = self.menuPos
        pos = self.viewBox().mapSceneToView(pos)
        self.chart.createNode(nodeType, pos=pos)

    def selectionChanged(self):
        items = self._scene.selectedItems()
        if len(items) == 0:
            data = None
        else:
            item = items[0]
            if hasattr(item, 'node') and isinstance(item.node, Node):
                n = item.node
                if n in self.ctrl.items:
                    self.ctrl.select(n)
                else:
                    self.ctrl.clearSelection()
                data = {'outputs': n.outputValues(), 'inputs': n.inputValues()}
                self.selNameLabel.setText(n.name())
                if hasattr(n, 'nodeName'):
                    self.selDescLabel.setText('<b>%s</b>: %s' % (n.nodeName, n.__class__.__doc__))
                else:
                    self.selDescLabel.setText('')
                if n.exception is not None:
                    data['exception'] = n.exception
            else:
                data = None
        self.selectedTree.setData(data, hideRoot=True)

    def hoverOver(self, items):
        term = None
        for item in items:
            if item is self.hoverItem:
                return
            self.hoverItem = item
            if hasattr(item, 'term') and isinstance(item.term, Terminal):
                term = item.term
                break
        if term is None:
            self.hoverText.setPlainText('')
        else:
            val = term.value()
            if isinstance(val, ndarray):
                val = '%s %s %s' % (type(val).__name__, str(val.shape), str(val.dtype))
            else:
                val = str(val)
                if len(val) > 400:
                    val = val[:400] + '...'
            self.hoverText.setPlainText('%s.%s = %s' % (term.node().name(), term.name(), val))

    def clear(self):
        self.selectedTree.setData(None)
        self.hoverText.setPlainText('')
        self.selNameLabel.setText('')
        self.selDescLabel.setText('')