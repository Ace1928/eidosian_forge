from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
import atexit
import enum
import mmap
import os
import sys
import tempfile
from .. import Qt
from .. import CONFIG_OPTIONS
from .. import multiprocess as mp
from .GraphicsView import GraphicsView
Close the remote process. After this call, the widget will no longer be updated.