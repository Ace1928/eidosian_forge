import operator
import weakref
from collections import OrderedDict
from functools import reduce
from math import hypot
from typing import Optional
from xml.etree.ElementTree import Element
from .. import functions as fn
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QtCore, QtWidgets, isQObjectAlive
def _replaceView(self, oldView, item=None):
    if item is None:
        item = self
    for child in item.childItems():
        if isinstance(child, GraphicsItem):
            if child.getViewBox() is oldView:
                child._updateView()
        else:
            self._replaceView(oldView, child)