import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class QtWidgets(ProxyNamespace):

    class QApplication(QtCore.QObject):

        def translate(uiname, text, disambig, encoding):
            return i18n_string(text or '', disambig)
        translate = staticmethod(translate)

    class QSpacerItem(ProxyClass):
        pass

    class QSizePolicy(ProxyClass):
        pass

    class QAction(QtCore.QObject):
        pass

    class QActionGroup(QtCore.QObject):
        pass

    class QButtonGroup(QtCore.QObject):
        pass

    class QLayout(QtCore.QObject):

        def setMargin(self, v):
            ProxyClassMember(self, 'setContentsMargins', 0)(v, v, v, v)

    class QGridLayout(QLayout):
        pass

    class QBoxLayout(QLayout):
        pass

    class QHBoxLayout(QBoxLayout):
        pass

    class QVBoxLayout(QBoxLayout):
        pass

    class QFormLayout(QLayout):
        pass

    class QWidget(QtCore.QObject):

        def font(self):
            return Literal('%s.font()' % self)

        def minimumSizeHint(self):
            return Literal('%s.minimumSizeHint()' % self)

        def sizePolicy(self):
            sp = LiteralProxyClass()
            sp._uic_name = '%s.sizePolicy()' % self
            return sp

    class QDialog(QWidget):
        pass

    class QWizard(QDialog):
        pass

    class QAbstractSlider(QWidget):
        pass

    class QDial(QAbstractSlider):
        pass

    class QScrollBar(QAbstractSlider):
        pass

    class QSlider(QAbstractSlider):
        pass

    class QMenu(QWidget):

        def menuAction(self):
            return Literal('%s.menuAction()' % self)

    class QTabWidget(QWidget):

        def addTab(self, *args):
            text = args[-1]
            if isinstance(text, i18n_string):
                i18n_print('%s.setTabText(%s.indexOf(%s), %s)' % (self._uic_name, self._uic_name, args[0], text))
                args = args[:-1] + ('',)
            ProxyClassMember(self, 'addTab', 0)(*args)

        def indexOf(self, page):
            return Literal('%s.indexOf(%s)' % (self, page))

    class QComboBox(QWidget):
        pass

    class QFontComboBox(QComboBox):
        pass

    class QAbstractSpinBox(QWidget):
        pass

    class QDoubleSpinBox(QAbstractSpinBox):
        pass

    class QSpinBox(QAbstractSpinBox):
        pass

    class QDateTimeEdit(QAbstractSpinBox):
        pass

    class QDateEdit(QDateTimeEdit):
        pass

    class QTimeEdit(QDateTimeEdit):
        pass

    class QFrame(QWidget):
        pass

    class QLabel(QFrame):
        pass

    class QLCDNumber(QFrame):
        pass

    class QSplitter(QFrame):
        pass

    class QStackedWidget(QFrame):
        pass

    class QToolBox(QFrame):

        def addItem(self, *args):
            text = args[-1]
            if isinstance(text, i18n_string):
                i18n_print('%s.setItemText(%s.indexOf(%s), %s)' % (self._uic_name, self._uic_name, args[0], text))
                args = args[:-1] + ('',)
            ProxyClassMember(self, 'addItem', 0)(*args)

        def indexOf(self, page):
            return Literal('%s.indexOf(%s)' % (self, page))

        def layout(self):
            return QtWidgets.QLayout('%s.layout()' % self, False, (), noInstantiation=True)

    class QAbstractScrollArea(QFrame):
        pass

    class QGraphicsView(QAbstractScrollArea):
        pass

    class QMdiArea(QAbstractScrollArea):
        pass

    class QPlainTextEdit(QAbstractScrollArea):
        pass

    class QScrollArea(QAbstractScrollArea):
        pass

    class QTextEdit(QAbstractScrollArea):
        pass

    class QTextBrowser(QTextEdit):
        pass

    class QAbstractItemView(QAbstractScrollArea):
        pass

    class QColumnView(QAbstractItemView):
        pass

    class QHeaderView(QAbstractItemView):
        pass

    class QListView(QAbstractItemView):
        pass

    class QTableView(QAbstractItemView):

        def horizontalHeader(self):
            return QtWidgets.QHeaderView('%s.horizontalHeader()' % self, False, (), noInstantiation=True)

        def verticalHeader(self):
            return QtWidgets.QHeaderView('%s.verticalHeader()' % self, False, (), noInstantiation=True)

    class QTreeView(QAbstractItemView):

        def header(self):
            return QtWidgets.QHeaderView('%s.header()' % self, False, (), noInstantiation=True)

    class QListWidgetItem(ProxyClass):
        pass

    class QListWidget(QListView):
        isSortingEnabled = i18n_func('isSortingEnabled')
        setSortingEnabled = i18n_void_func('setSortingEnabled')

        def item(self, row):
            return QtWidgets.QListWidgetItem('%s.item(%i)' % (self, row), False, (), noInstantiation=True)

    class QTableWidgetItem(ProxyClass):
        pass

    class QTableWidget(QTableView):
        isSortingEnabled = i18n_func('isSortingEnabled')
        setSortingEnabled = i18n_void_func('setSortingEnabled')

        def item(self, row, col):
            return QtWidgets.QTableWidgetItem('%s.item(%i, %i)' % (self, row, col), False, (), noInstantiation=True)

        def horizontalHeaderItem(self, col):
            return QtWidgets.QTableWidgetItem('%s.horizontalHeaderItem(%i)' % (self, col), False, (), noInstantiation=True)

        def verticalHeaderItem(self, row):
            return QtWidgets.QTableWidgetItem('%s.verticalHeaderItem(%i)' % (self, row), False, (), noInstantiation=True)

    class QTreeWidgetItem(ProxyClass):

        def child(self, index):
            return QtWidgets.QTreeWidgetItem('%s.child(%i)' % (self, index), False, (), noInstantiation=True)

    class QTreeWidget(QTreeView):
        isSortingEnabled = i18n_func('isSortingEnabled')
        setSortingEnabled = i18n_void_func('setSortingEnabled')

        def headerItem(self):
            return QtWidgets.QWidget('%s.headerItem()' % self, False, (), noInstantiation=True)

        def topLevelItem(self, index):
            return QtWidgets.QTreeWidgetItem('%s.topLevelItem(%i)' % (self, index), False, (), noInstantiation=True)

    class QAbstractButton(QWidget):
        pass

    class QCheckBox(QAbstractButton):
        pass

    class QRadioButton(QAbstractButton):
        pass

    class QToolButton(QAbstractButton):
        pass

    class QPushButton(QAbstractButton):
        pass

    class QCommandLinkButton(QPushButton):
        pass
    for _class in _qwidgets:
        if _class not in locals():
            locals()[_class] = type(_class, (QWidget,), {})