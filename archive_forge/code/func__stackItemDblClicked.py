import os
import sys
import pickle
import subprocess
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets
from .repl_widget import ReplWidget
from .exception_widget import ExceptionHandlerWidget
def _stackItemDblClicked(self, handler, item):
    editor = self.editor
    if editor is None:
        editor = getConfigOption('editorCommand')
    if editor is None:
        return
    tb = self.excHandler.selectedFrame()
    lineNum = tb.f_lineno
    fileName = tb.f_code.co_filename
    subprocess.Popen(self.editor.format(fileName=fileName, lineNum=lineNum), shell=True)