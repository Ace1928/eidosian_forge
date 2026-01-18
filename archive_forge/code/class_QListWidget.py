import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QListWidget(QListView):
    isSortingEnabled = i18n_func('isSortingEnabled')
    setSortingEnabled = i18n_void_func('setSortingEnabled')

    def item(self, row):
        return QtWidgets.QListWidgetItem('%s.item(%i)' % (self, row), False, (), noInstantiation=True)