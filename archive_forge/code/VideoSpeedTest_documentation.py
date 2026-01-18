import argparse
import itertools
import sys
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import VideoTemplate_generic as ui_template

Tests the speed of image updates for an ImageItem and RawImageWidget.
The speed will generally depend on the type of data being shown, whether
it is being scaled and/or converted by lookup table, and whether OpenGL
is used by the view widget
