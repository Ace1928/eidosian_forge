import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataModel
import pyomo.contrib.viewer.qt as myqt
from pyomo.common.dependencies import DeferredImportError
class UIData(object):
    model = None

    def __init__(*args, **kwargs):
        pass