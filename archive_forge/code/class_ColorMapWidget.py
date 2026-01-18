from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
class ColorMapWidget(ptree.ParameterTree):
    """
    This class provides a widget allowing the user to customize color mapping
    for multi-column data. Given a list of field names, the user may specify
    multiple criteria for assigning colors to each record in a numpy record array.
    Multiple criteria are evaluated and combined into a single color for each
    record by user-defined compositing methods.
    
    For simpler color mapping using a single gradient editor, see 
    :class:`GradientWidget <pyqtgraph.GradientWidget>`
    """
    sigColorMapChanged = QtCore.Signal(object)

    def __init__(self, parent=None):
        ptree.ParameterTree.__init__(self, parent=parent, showHeader=False)
        self.params = ColorMapParameter()
        self.setParameters(self.params)
        self.params.sigTreeStateChanged.connect(self.mapChanged)
        self.setFields = self.params.setFields
        self.map = self.params.map

    def mapChanged(self):
        self.sigColorMapChanged.emit(self)

    def widgetGroupInterface(self):
        return (self.sigColorMapChanged, self.saveState, self.restoreState)

    def saveState(self):
        return self.params.saveState()

    def restoreState(self, state):
        self.params.restoreState(state)

    def addColorMap(self, name):
        """Add a new color mapping and return the created parameter.
        """
        return self.params.addNew(name)