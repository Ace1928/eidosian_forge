import numpy as np
from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import CtrlNode

        Specify the set of plots (PlotWidget or PlotItem) that the user may
        select from.
        
        *plots* must be a dictionary of {name: plot} pairs.
        