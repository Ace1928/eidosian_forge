import collections
from collections.abc import Mapping as collections_Mapping
from pyomo.common.autoslots import AutoSlots
@staticmethod
def _unhashable(val):
    return id(val)