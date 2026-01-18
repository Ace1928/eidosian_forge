from numbers import Number
import numpy as np
import param
from ..core import Dimension, Element, Element2D
from ..core.data import Dataset
from ..core.util import datetime_types
class HSpans(VectorizedAnnotation):
    kdims = param.List(default=[Dimension('y0'), Dimension('y1')], bounds=(2, 2))
    group = param.String(default='HSpans', constant=True)