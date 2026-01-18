import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
def _set_value_callback(self, val):
    if isinstance(self.data, (Var._ComponentDataClass, BooleanVar._ComponentDataClass)):
        try:
            self.data.value = val
        except:
            return
    elif isinstance(self.data, (Var, BooleanVar)):
        try:
            for o in self.data.values():
                o.value = val
        except:
            return
    elif isinstance(self.data, _ParamData):
        if not self.data.parent_component().mutable:
            return
        try:
            self.data.value = val
        except:
            return
    elif isinstance(self.data, Param):
        if not self.data.parent_component().mutable:
            return
        try:
            for o in self.data.values():
                o.value = val
        except:
            return