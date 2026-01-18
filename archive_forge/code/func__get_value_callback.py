import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
def _get_value_callback(self):
    if isinstance(self.data, _ParamData):
        v = value_no_exception(self.data, div0='divide_by_0')
        if isinstance(v, float):
            v = float(v)
        elif isinstance(v, int):
            v = int(v)
        return v
    elif isinstance(self.data, (Var._ComponentDataClass, BooleanVar._ComponentDataClass)):
        v = value_no_exception(self.data)
        if isinstance(v, float):
            v = float(v)
        elif isinstance(v, int):
            v = int(v)
        return v
    elif isinstance(self.data, (float, int)):
        return self.data
    else:
        return self._cache_value