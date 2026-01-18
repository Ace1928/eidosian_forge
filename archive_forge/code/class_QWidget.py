import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QWidget(QtCore.QObject):

    def font(self):
        return Literal('%s.font()' % self)

    def minimumSizeHint(self):
        return Literal('%s.minimumSizeHint()' % self)

    def sizePolicy(self):
        sp = LiteralProxyClass()
        sp._uic_name = '%s.sizePolicy()' % self
        return sp