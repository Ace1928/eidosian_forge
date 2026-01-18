import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
class DynamicNames(object):
    """Fields names/positions mapping for component-less objects"""

    def __init__(self):
        self._keyToIdxMap = {}
        self._idxToKeyMap = {}

    def __len__(self):
        return len(self._keyToIdxMap)

    def __contains__(self, item):
        return item in self._keyToIdxMap or item in self._idxToKeyMap

    def __iter__(self):
        return (self._idxToKeyMap[idx] for idx in range(len(self._idxToKeyMap)))

    def __getitem__(self, item):
        try:
            return self._keyToIdxMap[item]
        except KeyError:
            return self._idxToKeyMap[item]

    def getNameByPosition(self, idx):
        try:
            return self._idxToKeyMap[idx]
        except KeyError:
            raise error.PyAsn1Error('Type position out of range')

    def getPositionByName(self, name):
        try:
            return self._keyToIdxMap[name]
        except KeyError:
            raise error.PyAsn1Error('Name %s not found' % (name,))

    def addField(self, idx):
        self._keyToIdxMap['field-%d' % idx] = idx
        self._idxToKeyMap[idx] = 'field-%d' % idx