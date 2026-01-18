import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
class flatten_stack(Operation):
    """
    Thin wrapper around datashader's shade operation to flatten
    ImageStacks into RGB elements.

    Used for the MPL and Plotly backends because these backends
    do not natively support ImageStacks, unlike Bokeh.
    """
    shade_params = param.Dict(default={}, doc="\n        Additional parameters passed to datashader's shade operation.")

    def _process(self, element, key=None):
        try:
            from ..operation.datashader import shade
        except ImportError as exc:
            raise ImportError('Flattening ImageStacks requires datashader.') from exc
        return shade(element, **self.shade_params)