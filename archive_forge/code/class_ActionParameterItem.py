from ...Qt import QtCore, QtWidgets, QtGui
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
class ActionParameterItem(ParameterItem):
    """ParameterItem displaying a clickable button."""

    def __init__(self, param, depth):
        ParameterItem.__init__(self, param, depth)
        self.layoutWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layoutWidget.setLayout(self.layout)
        self.button = ParameterControlledButton(param, self.layoutWidget)
        self.layout.addWidget(self.button)
        self.layout.addStretch()
        self.titleChanged()

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tree = self.treeWidget()
        if tree is None:
            return
        self.setFirstColumnSpanned(True)
        tree.setItemWidget(self, 0, self.layoutWidget)

    def titleChanged(self):
        self.setSizeHint(0, self.button.sizeHint())