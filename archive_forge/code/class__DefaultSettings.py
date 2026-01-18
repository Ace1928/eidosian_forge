import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from types import FunctionType
from IPython.display import display
from numbers import Integral
from traitlets import (
from itertools import chain
from uuid import uuid4
from six import string_types
from distutils.version import LooseVersion
class _DefaultSettings(object):

    def __init__(self):
        self._grid_options = {'fullWidthRows': True, 'syncColumnCellResize': True, 'forceFitColumns': True, 'defaultColumnWidth': 150, 'rowHeight': 28, 'enableColumnReorder': False, 'enableTextSelectionOnCells': True, 'editable': True, 'autoEdit': False, 'explicitInitialization': True, 'maxVisibleRows': 15, 'minVisibleRows': 8, 'sortable': True, 'filterable': True, 'highlightSelectedCell': False, 'highlightSelectedRow': True, 'boldIndex': True}
        self._column_options = {'editable': True, 'defaultSortAsc': True, 'maxWidth': None, 'minWidth': 30, 'resizable': True, 'sortable': True, 'toolTip': '', 'width': None}
        self._show_toolbar = False
        self._precision = None

    def set_grid_option(self, optname, optvalue):
        self._grid_options[optname] = optvalue

    def set_defaults(self, show_toolbar=None, precision=None, grid_options=None, column_options=None):
        if show_toolbar is not None:
            self._show_toolbar = show_toolbar
        if precision is not None:
            self._precision = precision
        if grid_options is not None:
            self._grid_options = grid_options
        if column_options is not None:
            self._column_options = column_options

    @property
    def show_toolbar(self):
        return self._show_toolbar

    @property
    def grid_options(self):
        return self._grid_options

    @property
    def precision(self):
        return self._precision or pd.get_option('display.precision') - 1

    @property
    def column_options(self):
        return self._column_options