import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QTreeWidgetItem(ProxyClass):

    def child(self, index):
        return QtWidgets.QTreeWidgetItem('%s.child(%i)' % (self, index), False, (), noInstantiation=True)