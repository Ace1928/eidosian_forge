import os
import sys
import pickle
import subprocess
from .. import getConfigOption
from ..Qt import QtCore, QtWidgets
from .repl_widget import ReplWidget
from .exception_widget import ExceptionHandlerWidget
def _commandEntered(self, repl, cmd):
    self.historyList.addItem(cmd)
    self.saveHistory(self.input.history[1:100])
    sb = self.historyList.verticalScrollBar()
    sb.setValue(sb.maximum())