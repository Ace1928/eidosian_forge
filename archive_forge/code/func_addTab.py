import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
def addTab(self, *args):
    text = args[-1]
    if isinstance(text, i18n_string):
        i18n_print('%s.setTabText(%s.indexOf(%s), %s)' % (self._uic_name, self._uic_name, args[0], text))
        args = args[:-1] + ('',)
    ProxyClassMember(self, 'addTab', 0)(*args)