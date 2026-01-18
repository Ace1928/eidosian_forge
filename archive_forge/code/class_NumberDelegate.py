import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
class NumberDelegate(myqt.QItemDelegate):
    """
    Tree view item delegate. This is used here to change how items are edited.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)
        factory = myqt.QItemEditorFactory()
        factory.registerEditor(myqt.QMetaType.Int, LineEditCreator())
        factory.registerEditor(myqt.QMetaType.Double, LineEditCreator())
        self.setItemEditorFactory(factory)

    def setModelData(self, editor, model, index):
        if isinstance(editor, myqt.QComboBox):
            value = editor.currentText()
        else:
            value = editor.text()
        a = model.column[index.column()]
        isinstance(index.internalPointer().get(a), bool)
        try:
            if value == 'False' or value == 'false':
                index.internalPointer().set(a, False)
            elif value == 'True' or value == 'true':
                index.internalPointer().set(a, True)
            elif '.' in value or 'e' in value or 'E' in value:
                index.internalPointer().set(a, float(value))
            else:
                index.internalPointer().set(a, int(value))
        except:
            pass