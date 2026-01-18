from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
class RangeFilterItem(ptree.types.SimpleParameter):

    def __init__(self, name, opts):
        self.fieldName = name
        units = opts.get('units', '')
        self.units = units
        ptree.types.SimpleParameter.__init__(self, name=name, autoIncrementName=True, type='bool', value=True, removable=True, renamable=True, children=[dict(name='Min', type='float', value=0.0, suffix=units, siPrefix=True), dict(name='Max', type='float', value=1.0, suffix=units, siPrefix=True)])

    def generateMask(self, data, mask):
        vals = data[self.fieldName][mask]
        mask[mask] = (vals >= self['Min']) & (vals < self['Max'])
        return mask

    def describe(self):
        return '%s < %s < %s' % (fn.siFormat(self['Min'], suffix=self.units), self.fieldName, fn.siFormat(self['Max'], suffix=self.units))

    def updateFilter(self, opts):
        pass