from __future__ import annotations
import typing as T
from .qt import QtBaseModule
from . import ModuleInfo
class Qt4Module(QtBaseModule):
    INFO = ModuleInfo('qt4')

    def __init__(self, interpreter: Interpreter):
        QtBaseModule.__init__(self, interpreter, qt_version=4)