import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QTableView(QAbstractItemView):

    def horizontalHeader(self):
        return QtWidgets.QHeaderView('%s.horizontalHeader()' % self, False, (), noInstantiation=True)

    def verticalHeader(self):
        return QtWidgets.QHeaderView('%s.verticalHeader()' % self, False, (), noInstantiation=True)