from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
def _setBounds(self, bounds):
    for line in self.lines:
        line.setBounds(bounds)