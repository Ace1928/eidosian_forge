from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
def addColorMap(self, name):
    """Add a new color mapping and return the created parameter.
        """
    return self.params.addNew(name)