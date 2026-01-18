import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
class TickMenu(QtWidgets.QMenu):

    def __init__(self, tick, sliderItem):
        QtWidgets.QMenu.__init__(self)
        self.tick = weakref.ref(tick)
        self.sliderItem = weakref.ref(sliderItem)
        self.removeAct = self.addAction(translate('GradientEditorItem', 'Remove Tick'), lambda: self.sliderItem().removeTick(tick))
        if not self.tick().removeAllowed or len(self.sliderItem().ticks) < 3:
            self.removeAct.setEnabled(False)
        positionMenu = self.addMenu(translate('GradientEditorItem', 'Set Position'))
        w = QtWidgets.QWidget()
        l = QtWidgets.QGridLayout()
        w.setLayout(l)
        value = sliderItem.tickValue(tick)
        self.fracPosSpin = SpinBox()
        self.fracPosSpin.setOpts(value=value, bounds=(0.0, 1.0), step=0.01, decimals=2)
        l.addWidget(QtWidgets.QLabel(f'{translate('GradiantEditorItem', 'Position')}:'), 0, 0)
        l.addWidget(self.fracPosSpin, 0, 1)
        a = QtWidgets.QWidgetAction(self)
        a.setDefaultWidget(w)
        positionMenu.addAction(a)
        self.fracPosSpin.sigValueChanging.connect(self.fractionalValueChanged)
        colorAct = self.addAction(translate('Context Menu', 'Set Color'), lambda: self.sliderItem().raiseColorDialog(self.tick()))
        if not self.tick().colorChangeAllowed:
            colorAct.setEnabled(False)

    def fractionalValueChanged(self, x):
        self.sliderItem().setTickValue(self.tick(), self.fracPosSpin.value())