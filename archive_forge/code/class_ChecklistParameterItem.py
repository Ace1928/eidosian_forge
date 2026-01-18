from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
class ChecklistParameterItem(GroupParameterItem):
    """
    Wraps a :class:`GroupParameterItem` to manage ``bool`` parameter children. Also provides convenience buttons to
    select or clear all values at once. Note these conveniences are disabled when ``exclusive`` is *True*.
    """

    def __init__(self, param, depth):
        self.btnGrp = QtWidgets.QButtonGroup()
        self.btnGrp.setExclusive(False)
        self._constructMetaBtns()
        super().__init__(param, depth)

    def _constructMetaBtns(self):
        self.metaBtnWidget = QtWidgets.QWidget()
        self.metaBtnLayout = lay = QtWidgets.QHBoxLayout(self.metaBtnWidget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.metaBtns = {}
        lay.addStretch(0)
        for title in ('Clear', 'Select'):
            self.metaBtns[title] = btn = QtWidgets.QPushButton(f'{title} All')
            self.metaBtnLayout.addWidget(btn)
            btn.clicked.connect(getattr(self, f'{title.lower()}AllClicked'))
        self.metaBtns['default'] = self.makeDefaultButton()
        self.metaBtnLayout.addWidget(self.metaBtns['default'])

    def treeWidgetChanged(self):
        ParameterItem.treeWidgetChanged(self)
        tw = self.treeWidget()
        if tw is None:
            return
        tw.setItemWidget(self, 1, self.metaBtnWidget)

    def selectAllClicked(self):
        self.param.valChangingProxy.timer.stop()
        self.param.setValue(self.param.reverse[0])

    def clearAllClicked(self):
        self.param.valChangingProxy.timer.stop()
        self.param.setValue([])

    def insertChild(self, pos, item):
        ret = super().insertChild(pos, item)
        self.btnGrp.addButton(item.widget)
        return ret

    def addChild(self, item):
        ret = super().addChild(item)
        self.btnGrp.addButton(item.widget)
        return ret

    def takeChild(self, i):
        child = super().takeChild(i)
        self.btnGrp.removeButton(child.widget)

    def optsChanged(self, param, opts):
        super().optsChanged(param, opts)
        if 'expanded' in opts:
            for btn in self.metaBtns.values():
                btn.setVisible(opts['expanded'])
        exclusive = opts.get('exclusive', param.opts['exclusive'])
        enabled = opts.get('enabled', param.opts['enabled'])
        for name, btn in self.metaBtns.items():
            if name != 'default':
                btn.setDisabled(exclusive or not enabled)
        self.btnGrp.setExclusive(exclusive)
        if 'limits' not in opts and ('enabled' in opts or 'readonly' in opts):
            self.updateDefaultBtn()

    def expandedChangedEvent(self, expanded):
        for btn in self.metaBtns.values():
            btn.setVisible(expanded)

    def valueChanged(self, param, val):
        self.updateDefaultBtn()

    def updateDefaultBtn(self):
        self.metaBtns['default'].setEnabled(not self.param.valueIsDefault() and self.param.opts['enabled'] and self.param.writable())
        return
    makeDefaultButton = WidgetParameterItem.makeDefaultButton
    defaultClicked = WidgetParameterItem.defaultClicked