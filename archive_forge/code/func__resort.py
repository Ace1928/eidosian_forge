from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
def _resort(self):
    self.data = dict(dimension_sort(self.data, self.kdims, self.vdims, range(self.ndims)))