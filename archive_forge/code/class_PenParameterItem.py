import re
from contextlib import ExitStack
from ... import functions as fn
from ...Qt import QtCore, QtWidgets
from ...SignalProxy import SignalProxy
from ...widgets.PenPreviewLabel import PenPreviewLabel
from . import GroupParameterItem, WidgetParameterItem
from .basetypes import GroupParameter, Parameter, ParameterItem
from .qtenum import QtEnumParameter
class PenParameterItem(GroupParameterItem):

    def __init__(self, param, depth):
        self.defaultBtn = self.makeDefaultButton()
        super().__init__(param, depth)
        self.itemWidget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.penLabel = PenPreviewLabel(param)
        for child in (self.penLabel, self.defaultBtn):
            layout.addWidget(child)
        self.itemWidget.setLayout(layout)

    def optsChanged(self, param, opts):
        if 'enabled' in opts or 'readonly' in opts:
            self.updateDefaultBtn()

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tw = self.treeWidget()
        if tw is None:
            return
        tw.setItemWidget(self, 1, self.itemWidget)
    defaultClicked = WidgetParameterItem.defaultClicked
    makeDefaultButton = WidgetParameterItem.makeDefaultButton

    def valueChanged(self, param, val):
        self.updateDefaultBtn()

    def updateDefaultBtn(self):
        self.defaultBtn.setEnabled(not self.param.valueIsDefault() and self.param.opts['enabled'] and self.param.writable())