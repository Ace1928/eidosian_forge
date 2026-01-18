from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
class ChecklistParameter(GroupParameter):
    """
    Can be set just like a :class:`ListParameter`, but allows for multiple values to be selected simultaneously.

    ============== ========================================================
    **Options**
    exclusive      When *False*, any number of options can be selected. The resulting ``value()`` is a list of
                   all checked values. When *True*, it behaves like a ``list`` type -- only one value can be selected.
                   If no values are selected and ``exclusive`` is set to *True*, the first available limit is selected.
                   The return value of an ``exclusive`` checklist is a single value rather than a list with one element.
    delay          Controls the wait time between editing the checkboxes/radio button children and firing a "value changed"
                   signal. This allows users to edit multiple boxes at once for a single value update.
    ============== ========================================================
    """
    itemClass = ChecklistParameterItem

    def __init__(self, **opts):
        if 'children' in opts:
            raise ValueError("Cannot pass 'children' to ChecklistParameter. Pass a 'value' key only.")
        self.targetValue = None
        limits = opts.setdefault('limits', [])
        self.forward, self.reverse = ListParameter.mapping(limits)
        value = opts.setdefault('value', limits)
        opts.setdefault('exclusive', False)
        super().__init__(**opts)
        self.sigLimitsChanged.connect(self.updateLimits)
        self.sigOptionsChanged.connect(self.optsChanged)
        if len(limits):
            self.updateLimits(self, limits)
            self.setValue(value)
        self.valChangingProxy = SignalProxy(self.sigValueChanging, delay=opts.get('delay', 1.0), slot=self._finishChildChanges, threadSafe=False)

    def childrenValue(self):
        vals = [self.forward[p.name()] for p in self.children() if p.value()]
        exclusive = self.opts['exclusive']
        if not vals and exclusive:
            return None
        elif exclusive:
            return vals[0]
        else:
            return vals

    def _onChildChanging(self, child, value):
        if self.opts['exclusive'] and value:
            value = self.forward[child.name()]
        else:
            value = self.childrenValue()
        self.sigValueChanging.emit(self, value)

    def updateLimits(self, _param, limits):
        oldOpts = self.names
        val = self.opts.get('value', None)
        self.blockTreeChangeSignal()
        self.clearChildren()
        self.forward, self.reverse = ListParameter.mapping(limits)
        if self.opts.get('exclusive'):
            typ = 'radio'
        else:
            typ = 'bool'
        for chName in self.forward:
            newVal = bool(oldOpts.get(chName, False))
            child = BoolOrRadioParameter(type=typ, name=chName, value=newVal, default=None)
            self.addChild(child)
            child.blockTreeChangeSignal()
            child.sigValueChanged.connect(self._onChildChanging)
        self.treeStateChanges.clear()
        self.unblockTreeChangeSignal()
        self.setValue(val)

    def _finishChildChanges(self, paramAndValue):
        param, value = paramAndValue
        return self.setValue(value)

    def optsChanged(self, param, opts):
        if 'exclusive' in opts:
            self.updateLimits(None, self.opts.get('limits', []))
        if 'delay' in opts:
            self.valChangingProxy.setDelay(opts['delay'])

    def setValue(self, value, blockSignal=None):
        self.targetValue = value
        if not isinstance(value, list):
            value = [value]
        names, values = self._intersectionWithLimits(value)
        valueToSet = values
        if self.opts['exclusive']:
            if len(self.forward):
                names.append(self.reverse[1][0])
            if len(names) > 1:
                names = names[:1]
            if not len(names):
                valueToSet = None
            else:
                valueToSet = self.forward[names[0]]
        for chParam in self:
            checked = chParam.name() in names
            chParam.setValue(checked, self._onChildChanging)
        super().setValue(valueToSet, blockSignal)

    def _intersectionWithLimits(self, values: list):
        """
        Returns the (names, values) from limits that intersect with ``values``.
        """
        allowedNames = []
        allowedValues = []
        for val in values:
            for limitName, limitValue in zip(*self.reverse):
                if fn.eq(limitValue, val):
                    allowedNames.append(limitName)
                    allowedValues.append(val)
                    break
        return (allowedNames, allowedValues)

    def setToDefault(self):
        self.valChangingProxy.timer.stop()
        super().setToDefault()

    def saveState(self, filter=None):
        state = super().saveState(filter)
        state.pop('children', None)
        return state

    def restoreState(self, state, recursive=True, addChildren=True, removeChildren=True, blockSignals=True):
        return super().restoreState(state, recursive, addChildren=False, removeChildren=False, blockSignals=blockSignals)