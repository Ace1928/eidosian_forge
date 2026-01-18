import numpy as np
import pandas as pd
from packaging.version import Version
from pandas.api.types import is_numeric_dtype
from .. import util
from ..dimension import Dimension, dimension_name
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .interface import DataError, Interface
from .util import finite_range
class PandasAPI:
    """
    This class is used to describe the interface as having a pandas-like API.

    The reason to have this class is that it is not always
    possible to directly inherit from the PandasInterface.

    This class should not have any logic as it should be used like:
        if issubclass(interface, PandasAPI):
            ...
    """