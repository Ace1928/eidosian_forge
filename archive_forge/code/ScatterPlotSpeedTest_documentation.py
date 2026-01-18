import argparse
import itertools
import re
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtWidgets

For testing rapid updates of ScatterPlotItem under various conditions.

(Scatter plots are still rather slow to draw; expect about 20fps)
