import sys
import warnings
from itertools import product
import numpy as np
import pandas as pd
from .. import util
from ..dimension import dimension_name
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .interface import DataError, Interface
from .pandas import PandasInterface
from .util import finite_range

        Given a Dataset object and a dictionary with dimension keys and
        selection keys (i.e. tuple ranges, slices, sets, lists, or literals)
        return a boolean mask over the rows in the Dataset object that
        have been selected.
        