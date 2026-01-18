import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def _renameChild(self, child, name):
    if name in self.names:
        return child.name()
    self.names[name] = child
    del self.names[child.name()]
    return name