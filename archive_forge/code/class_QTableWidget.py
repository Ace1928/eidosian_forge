import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QTableWidget(QTableView):
    isSortingEnabled = i18n_func('isSortingEnabled')
    setSortingEnabled = i18n_void_func('setSortingEnabled')

    def item(self, row, col):
        return QtWidgets.QTableWidgetItem('%s.item(%i, %i)' % (self, row, col), False, (), noInstantiation=True)

    def horizontalHeaderItem(self, col):
        return QtWidgets.QTableWidgetItem('%s.horizontalHeaderItem(%i)' % (self, col), False, (), noInstantiation=True)

    def verticalHeaderItem(self, row):
        return QtWidgets.QTableWidgetItem('%s.verticalHeaderItem(%i)' % (self, row), False, (), noInstantiation=True)