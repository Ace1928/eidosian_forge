import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
from pyomo.core.base.param import _ParamData
from pyomo.environ import (
from pyomo.common.fileutils import this_file_dir
def _add_root_item(self, o):
    """
        Add a root tree item
        """
    item = ComponentDataItem(None, o, ui_data=self.ui_data)
    self.rootItems.append(item)
    return item