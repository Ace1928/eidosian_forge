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
class FlowchartCtrlWidget(QtWidgets.QWidget):
    """The widget that contains the list of all the nodes in a flowchart and their controls, as well as buttons for loading/saving flowcharts."""

    def __init__(self, chart):
        self.items = {}
        self.currentFileName = None
        QtWidgets.QWidget.__init__(self)
        self.chart = chart
        self.ui = FlowchartCtrlTemplate.Ui_Form()
        self.ui.setupUi(self)
        self.ui.ctrlList.setColumnCount(2)
        self.ui.ctrlList.setColumnWidth(1, 20)
        self.ui.ctrlList.setVerticalScrollMode(self.ui.ctrlList.ScrollMode.ScrollPerPixel)
        self.ui.ctrlList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chartWidget = FlowchartWidget(chart, self)
        self.cwWin = QtWidgets.QMainWindow()
        self.cwWin.setWindowTitle('Flowchart')
        self.cwWin.setCentralWidget(self.chartWidget)
        self.cwWin.resize(1000, 800)
        h = self.ui.ctrlList.header()
        h.setSectionResizeMode(0, h.ResizeMode.Stretch)
        self.ui.ctrlList.itemChanged.connect(self.itemChanged)
        self.ui.loadBtn.clicked.connect(self.loadClicked)
        self.ui.saveBtn.clicked.connect(self.saveClicked)
        self.ui.saveAsBtn.clicked.connect(self.saveAsClicked)
        self.ui.showChartBtn.toggled.connect(self.chartToggled)
        self.chart.sigFileLoaded.connect(self.setCurrentFile)
        self.ui.reloadBtn.clicked.connect(self.reloadClicked)
        self.chart.sigFileSaved.connect(self.fileSaved)

    def chartToggled(self, b):
        if b:
            self.cwWin.show()
        else:
            self.cwWin.hide()

    def reloadClicked(self):
        try:
            self.chartWidget.reloadLibrary()
            self.ui.reloadBtn.success('Reloaded.')
        except:
            self.ui.reloadBtn.success('Error.')
            raise

    def loadClicked(self):
        self.chart.loadFile()

    def fileSaved(self, fileName):
        self.setCurrentFile(fileName)
        self.ui.saveBtn.success('Saved.')

    def saveClicked(self):
        if self.currentFileName is None:
            self.saveAsClicked()
        else:
            try:
                self.chart.saveFile(self.currentFileName)
            except:
                self.ui.saveBtn.failure('Error')
                raise

    def saveAsClicked(self):
        try:
            if self.currentFileName is None:
                self.chart.saveFile()
            else:
                self.chart.saveFile(suggestedFileName=self.currentFileName)
        except:
            self.ui.saveBtn.failure('Error')
            raise

    def setCurrentFile(self, fileName):
        self.currentFileName = fileName
        if fileName is None:
            self.ui.fileNameLabel.setText('<b>[ new ]</b>')
        else:
            self.ui.fileNameLabel.setText('<b>%s</b>' % os.path.split(self.currentFileName)[1])
        self.resizeEvent(None)

    def itemChanged(self, *args):
        pass

    def scene(self):
        return self.chartWidget.scene()

    def viewBox(self):
        return self.chartWidget.viewBox()

    def nodeRenamed(self, node, oldName):
        self.items[node].setText(0, node.name())

    def addNode(self, node):
        ctrl = node.ctrlWidget()
        item = QtWidgets.QTreeWidgetItem([node.name(), '', ''])
        self.ui.ctrlList.addTopLevelItem(item)
        byp = QtWidgets.QPushButton('X')
        byp.setCheckable(True)
        byp.setFixedWidth(20)
        item.bypassBtn = byp
        self.ui.ctrlList.setItemWidget(item, 1, byp)
        byp.node = node
        node.bypassButton = byp
        byp.setChecked(node.isBypassed())
        byp.clicked.connect(self.bypassClicked)
        if ctrl is not None:
            item2 = QtWidgets.QTreeWidgetItem()
            item.addChild(item2)
            self.ui.ctrlList.setItemWidget(item2, 0, ctrl)
        self.items[node] = item

    def removeNode(self, node):
        if node in self.items:
            item = self.items[node]
            try:
                item.bypassBtn.clicked.disconnect(self.bypassClicked)
            except (TypeError, RuntimeError):
                pass
            self.ui.ctrlList.removeTopLevelItem(item)

    def bypassClicked(self):
        btn = QtCore.QObject.sender(self)
        btn.node.bypass(btn.isChecked())

    def chartWidget(self):
        return self.chartWidget

    def outputChanged(self, data):
        pass

    def clear(self):
        self.chartWidget.clear()

    def select(self, node):
        item = self.items[node]
        self.ui.ctrlList.setCurrentItem(item)

    def clearSelection(self):
        self.ui.ctrlList.selectionModel().clearSelection()