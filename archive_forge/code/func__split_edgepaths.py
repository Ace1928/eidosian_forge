from collections import defaultdict
from types import FunctionType
import numpy as np
import pandas as pd
import param
from ..core import Dataset, Dimension, Element2D
from ..core.accessors import Redim
from ..core.operation import Operation
from ..core.util import is_dataframe, max_range, search_indices
from .chart import Points
from .path import Path
from .util import (
@property
def _split_edgepaths(self):
    if len(self) == len(self.edgepaths.data):
        return self.edgepaths
    else:
        return self.edgepaths.clone(split_path(self.edgepaths))