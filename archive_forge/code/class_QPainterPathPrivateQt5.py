import ctypes
import itertools
import numpy as np
from . import QT_LIB, QtCore, QtGui, compat
class QPainterPathPrivateQt5(ctypes.Structure):
    _fields_ = [('ref', ctypes.c_int), ('adata', ctypes.POINTER(QArrayDataQt5))]