from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
class EnumColorMapItem(ptree.types.GroupParameter):
    mapType = 'enum'

    def __init__(self, name, opts):
        self.fieldName = name
        vals = opts.get('values', [])
        if isinstance(vals, list):
            vals = OrderedDict([(v, str(v)) for v in vals])
        childs = []
        for val, vname in vals.items():
            ch = ptree.Parameter.create(name=vname, type='color')
            ch.maskValue = val
            childs.append(ch)
        ptree.types.GroupParameter.__init__(self, name=name, autoIncrementName=True, removable=True, renamable=True, children=[dict(name='Values', type='group', children=childs), dict(name='Operation', type='list', value='Overlay', limits=['Overlay', 'Add', 'Multiply', 'Set']), dict(name='Channels..', type='group', expanded=False, children=[dict(name='Red', type='bool', value=True), dict(name='Green', type='bool', value=True), dict(name='Blue', type='bool', value=True), dict(name='Alpha', type='bool', value=True)]), dict(name='Enabled', type='bool', value=True), dict(name='Default', type='color')])

    def map(self, data):
        data = data[self.fieldName]
        colors = np.empty((len(data), 4))
        default = np.array(self['Default'].getRgbF())
        colors[:] = default
        for v in self.param('Values'):
            mask = data == v.maskValue
            c = np.array(v.value().getRgbF())
            colors[mask] = c
        return colors