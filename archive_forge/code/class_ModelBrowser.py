import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
class ModelBrowser(_ModelBrowser, _ModelBrowserUI):

    def __init__(self, ui_data, standard='Var'):
        """
        Create a dock widget with a QTreeView of a Pyomo model.

        Args:
            ui_data: Contains model and ui information
            standard: A standard setup for different types of model components
                {"Var", "Constraint", "Param", "Expression"}
        """
        super().__init__()
        self.setupUi(self)
        number_delegate = NumberDelegate(self)
        self.ui_data = ui_data
        self.ui_data.updated.connect(self.update_model)
        self.treeView.setItemDelegate(number_delegate)
        if standard == 'Var':
            components = (Var, BooleanVar)
            columns = ['name', 'value', 'ub', 'lb', 'fixed', 'stale', 'units', 'domain']
            editable = ['value', 'ub', 'lb', 'fixed']
            self.setWindowTitle('Variables')
        elif standard == 'Constraint':
            components = Constraint
            columns = ['name', 'value', 'ub', 'lb', 'residual', 'active']
            editable = ['active']
            self.setWindowTitle('Constraints')
        elif standard == 'Param':
            components = Param
            columns = ['name', 'value', 'mutable', 'units']
            editable = ['value']
            self.setWindowTitle('Parameters')
        elif standard == 'Expression':
            components = Expression
            columns = ['name', 'value', 'units']
            editable = []
            self.setWindowTitle('Expressions')
        else:
            raise ValueError('{} is not a valid view type'.format(standard))
        datmodel = ComponentDataModel(self, ui_data=ui_data, columns=columns, components=components, editable=editable)
        self.datmodel = datmodel
        self.treeView.setModel(datmodel)
        self.treeView.setColumnWidth(0, 400)
        self.treeView.setSelectionBehavior(myqt.QAbstractItemView.SelectRows)
        self.treeView.setSelectionMode(myqt.QAbstractItemView.ExtendedSelection)

    def refresh(self):
        added = self.datmodel._update_tree()
        self.datmodel.layoutChanged.emit()

    def update_model(self):
        self.datmodel.update_model()