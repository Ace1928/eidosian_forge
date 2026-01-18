from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
class FormWidget(QtWidgets.QWidget):
    update_buttons = QtCore.Signal()

    def __init__(self, data, comment='', with_margin=False, parent=None):
        """
        Parameters
        ----------
        data : list of (label, value) pairs
            The data to be edited in the form.
        comment : str, optional
        with_margin : bool, default: False
            If False, the form elements reach to the border of the widget.
            This is the desired behavior if the FormWidget is used as a widget
            alongside with other widgets such as a QComboBox, which also do
            not have a margin around them.
            However, a margin can be desired if the FormWidget is the only
            widget within a container, e.g. a tab in a QTabWidget.
        parent : QWidget or None
            The parent widget.
        """
        super().__init__(parent)
        self.data = copy.deepcopy(data)
        self.widgets = []
        self.formlayout = QtWidgets.QFormLayout(self)
        if not with_margin:
            self.formlayout.setContentsMargins(0, 0, 0, 0)
        if comment:
            self.formlayout.addRow(QtWidgets.QLabel(comment))
            self.formlayout.addRow(QtWidgets.QLabel(' '))

    def get_dialog(self):
        """Return FormDialog instance"""
        dialog = self.parent()
        while not isinstance(dialog, QtWidgets.QDialog):
            dialog = dialog.parent()
        return dialog

    def setup(self):
        for label, value in self.data:
            if label is None and value is None:
                self.formlayout.addRow(QtWidgets.QLabel(' '), QtWidgets.QLabel(' '))
                self.widgets.append(None)
                continue
            elif label is None:
                self.formlayout.addRow(QtWidgets.QLabel(value))
                self.widgets.append(None)
                continue
            elif tuple_to_qfont(value) is not None:
                field = FontLayout(value, self)
            elif label.lower() not in BLACKLIST and mcolors.is_color_like(value):
                field = ColorLayout(to_qcolor(value), self)
            elif isinstance(value, str):
                field = QtWidgets.QLineEdit(value, self)
            elif isinstance(value, (list, tuple)):
                if isinstance(value, tuple):
                    value = list(value)
                selindex = value.pop(0)
                field = QtWidgets.QComboBox(self)
                if isinstance(value[0], (list, tuple)):
                    keys = [key for key, _val in value]
                    value = [val for _key, val in value]
                else:
                    keys = value
                field.addItems(value)
                if selindex in value:
                    selindex = value.index(selindex)
                elif selindex in keys:
                    selindex = keys.index(selindex)
                elif not isinstance(selindex, Integral):
                    _log.warning("index '%s' is invalid (label: %s, value: %s)", selindex, label, value)
                    selindex = 0
                field.setCurrentIndex(selindex)
            elif isinstance(value, bool):
                field = QtWidgets.QCheckBox(self)
                field.setChecked(value)
            elif isinstance(value, Integral):
                field = QtWidgets.QSpinBox(self)
                field.setRange(-10 ** 9, 10 ** 9)
                field.setValue(value)
            elif isinstance(value, Real):
                field = QtWidgets.QLineEdit(repr(value), self)
                field.setCursorPosition(0)
                field.setValidator(QtGui.QDoubleValidator(field))
                field.validator().setLocale(QtCore.QLocale('C'))
                dialog = self.get_dialog()
                dialog.register_float_field(field)
                field.textChanged.connect(lambda text: dialog.update_buttons())
            elif isinstance(value, datetime.datetime):
                field = QtWidgets.QDateTimeEdit(self)
                field.setDateTime(value)
            elif isinstance(value, datetime.date):
                field = QtWidgets.QDateEdit(self)
                field.setDate(value)
            else:
                field = QtWidgets.QLineEdit(repr(value), self)
            self.formlayout.addRow(label, field)
            self.widgets.append(field)

    def get(self):
        valuelist = []
        for index, (label, value) in enumerate(self.data):
            field = self.widgets[index]
            if label is None:
                continue
            elif tuple_to_qfont(value) is not None:
                value = field.get_font()
            elif isinstance(value, str) or mcolors.is_color_like(value):
                value = str(field.text())
            elif isinstance(value, (list, tuple)):
                index = int(field.currentIndex())
                if isinstance(value[0], (list, tuple)):
                    value = value[index][0]
                else:
                    value = value[index]
            elif isinstance(value, bool):
                value = field.isChecked()
            elif isinstance(value, Integral):
                value = int(field.value())
            elif isinstance(value, Real):
                value = float(str(field.text()))
            elif isinstance(value, datetime.datetime):
                datetime_ = field.dateTime()
                if hasattr(datetime_, 'toPyDateTime'):
                    value = datetime_.toPyDateTime()
                else:
                    value = datetime_.toPython()
            elif isinstance(value, datetime.date):
                date_ = field.date()
                if hasattr(date_, 'toPyDate'):
                    value = date_.toPyDate()
                else:
                    value = date_.toPython()
            else:
                value = literal_eval(str(field.text()))
            valuelist.append(value)
        return valuelist