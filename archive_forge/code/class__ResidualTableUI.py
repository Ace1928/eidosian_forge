import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
class _ResidualTableUI(object):
    pass