import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
def _set_lb_callback(self, val):
    if isinstance(self.data, Var._ComponentDataClass):
        try:
            self.data.setlb(val)
        except:
            return
    elif isinstance(self.data, Var):
        try:
            for o in self.data.values():
                o.setlb(val)
        except:
            return