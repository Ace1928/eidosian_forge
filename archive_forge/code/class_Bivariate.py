import numpy as np
import param
from ..core.data import Dataset
from ..core.dimension import Dimension, process_dimensions
from ..core.element import Element, Element2D
from ..core.util import get_param_values, unique_iterator
from .selection import Selection1DExpr, Selection2DExpr
class Bivariate(Selection2DExpr, StatisticsElement):
    """
    Bivariate elements are containers for two dimensional data, which
    is to be visualized as a kernel density estimate. The data should
    be supplied in a tabular format of x- and y-columns.
    """
    group = param.String(default='Bivariate', constant=True)
    kdims = param.List(default=[Dimension('x'), Dimension('y')], bounds=(2, 2))
    vdims = param.List(default=[Dimension('Density')], bounds=(0, 1))