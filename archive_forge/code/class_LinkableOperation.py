import numpy as np
import param
from param.parameterized import bothmethod
from ..core import Dataset, Operation
from ..core.util import datetime_types, dt_to_int, isfinite, max_range
from ..element import Image
from ..streams import PlotSize, RangeX, RangeXY
class LinkableOperation(Operation):
    """
    Abstract baseclass for operations supporting linked inputs.
    """
    link_inputs = param.Boolean(default=True, doc='\n        By default, the link_inputs parameter is set to True so that\n        when applying an operation, backends that support linked\n        streams update RangeXY streams on the inputs of the operation.\n        Disable when you do not want the resulting plot to be\n        interactive, e.g. when trying to display an interactive plot a\n        second time.')
    _allow_extra_keywords = True