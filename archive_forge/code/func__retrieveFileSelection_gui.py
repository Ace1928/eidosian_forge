import os
import re
from ...Qt import QtCore, QtGui, QtWidgets
from ..Parameter import Parameter
from .str import StrParameterItem
def _retrieveFileSelection_gui(self):
    curVal = self.param.value() if self.param.hasValue() else None
    if isinstance(curVal, list) and len(curVal):
        curVal = curVal[0]
        if os.path.isfile(curVal):
            curVal = os.path.dirname(curVal)
    opts = self.param.opts.copy()
    useDir = curVal or opts.get('directory') or os.getcwd()
    startDir = os.path.abspath(useDir)
    if os.path.isfile(startDir):
        opts['selectFile'] = os.path.basename(startDir)
        startDir = os.path.dirname(startDir)
    if os.path.exists(startDir):
        opts['directory'] = startDir
    if opts.get('windowTitle') is None:
        opts['windowTitle'] = self.param.title()
    if (fname := popupFilePicker(None, **opts)):
        self.param.setValue(fname)