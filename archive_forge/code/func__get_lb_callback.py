import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
def _get_lb_callback(self):
    if isinstance(self.data, Var._ComponentDataClass):
        return self.data.lb
    elif hasattr(self.data, 'lower'):
        return value_no_exception(self.data.lower, div0='Divide_by_0')
    else:
        return None