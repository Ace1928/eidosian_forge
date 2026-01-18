from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
class RangeColorMapItem(ptree.types.ColorMapParameter):
    mapType = 'range'

    def __init__(self, name, opts):
        self.fieldName = name
        units = opts.get('units', '')
        ptree.types.ColorMapParameter.__init__(self, name=name, autoIncrementName=True, type='colormap', removable=True, renamable=True, children=[dict(name='Min', type='float', value=0.0, suffix=units, siPrefix=True), dict(name='Max', type='float', value=1.0, suffix=units, siPrefix=True), dict(name='Operation', type='list', value='Overlay', limits=['Overlay', 'Add', 'Multiply', 'Set']), dict(name='Channels..', type='group', expanded=False, children=[dict(name='Red', type='bool', value=True), dict(name='Green', type='bool', value=True), dict(name='Blue', type='bool', value=True), dict(name='Alpha', type='bool', value=True)]), dict(name='Enabled', type='bool', value=True), dict(name='NaN', type='color')])

    def map(self, data):
        data = data[self.fieldName]
        scaled = fn.clip_array((data - self['Min']) / (self['Max'] - self['Min']), 0, 1)
        cmap = self.value()
        colors = cmap.map(scaled, mode='float')
        mask = np.invert(np.isfinite(data))
        nanColor = self['NaN']
        nanColor = nanColor.getRgbF()
        colors[mask] = nanColor
        return colors