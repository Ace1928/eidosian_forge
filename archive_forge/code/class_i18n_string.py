import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class i18n_string(object):

    def __init__(self, string, disambig):
        self.string = string
        self.disambig = disambig

    def __str__(self):
        if self.disambig is None:
            disambig = 'None'
        else:
            disambig = as_string(self.disambig, encode=False)
        return 'QtWidgets.QApplication.translate("%s", %s, %s, -1)' % (i18n_context, as_string(self.string, encode=False), disambig)