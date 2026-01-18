import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def _applyMenuEnabled(self):
    enableMenu = self.state.get('enableMenu', True)
    if enableMenu and self.menu is None:
        self.menu = ViewBoxMenu(self)
        self.updateViewLists()
    elif not enableMenu and self.menu is not None:
        self.menu.setParent(None)
        self.menu = None