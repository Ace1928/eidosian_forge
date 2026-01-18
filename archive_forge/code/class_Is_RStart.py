import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
class Is_RStart(Protocol):

    @property
    def rhome(self):
        ...

    @rhome.setter
    def rhome(self, value) -> None:
        ...

    @property
    def home(self):
        ...

    @home.setter
    def home(self, value) -> None:
        ...

    @property
    def CharacterMode(self):
        ...

    @CharacterMode.setter
    def CharacterMode(self, value) -> None:
        ...

    @property
    def ReadConsole(self):
        ...

    @ReadConsole.setter
    def ReadConsole(self, value) -> None:
        ...

    @property
    def WriteConsoleEx(self):
        ...

    @WriteConsoleEx.setter
    def WriteConsoleEx(self, value) -> None:
        ...

    @property
    def CallBack(self):
        ...

    @CallBack.setter
    def CallBack(self, value) -> None:
        ...

    @property
    def ShowMessage(self):
        ...

    @ShowMessage.setter
    def ShowMessage(self, value) -> None:
        ...

    @property
    def YesNoCancel(self):
        ...

    @YesNoCancel.setter
    def YesNoCancel(self, value) -> None:
        ...

    @property
    def Busy(self):
        ...

    @Busy.setter
    def Busy(self, value) -> None:
        ...

    @property
    def R_Quiet(self):
        ...

    @R_Quiet.setter
    def R_Quiet(self, value) -> None:
        ...

    @property
    def R_Interactive(self):
        ...

    @R_Interactive.setter
    def R_Interactive(self, value) -> None:
        ...

    @property
    def RestoreAction(self):
        ...

    @RestoreAction.setter
    def RestoreAction(self, value) -> None:
        ...

    @property
    def SaveAction(self):
        ...

    @SaveAction.setter
    def SaveAction(self, value) -> None:
        ...

    @property
    def vsize(self):
        ...

    @vsize.setter
    def vsize(self, value) -> None:
        ...

    @property
    def nsize(self):
        ...

    @nsize.setter
    def nsize(self, value) -> None:
        ...

    @property
    def max_vsize(self):
        ...

    @max_vsize.setter
    def max_vsize(self, value) -> None:
        ...

    @property
    def max_nsize(self):
        ...

    @max_nsize.setter
    def max_nsize(self, value) -> None:
        ...

    @property
    def ppsize(self):
        ...

    @ppsize.setter
    def ppsize(self, value) -> None:
        ...