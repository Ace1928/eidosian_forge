import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
def _get_units_callback(self):
    if isinstance(self.data, (Var, Var._ComponentDataClass)):
        return str(units.get_units(self.data))
    if isinstance(self.data, (Param, _ParamData)):
        return str(units.get_units(self.data))
    return self._cache_units