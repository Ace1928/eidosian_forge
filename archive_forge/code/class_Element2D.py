from itertools import groupby
import numpy as np
import pandas as pd
import param
from .dimension import Dimensioned, ViewableElement, asdim
from .layout import Composable, Layout, NdLayout
from .ndmapping import NdMapping
from .overlay import CompositeOverlay, NdOverlay, Overlayable
from .spaces import GridSpace, HoloMap
from .tree import AttrTree
from .util import get_param_values
class Element2D(Element):
    extents = param.Tuple(default=(None, None, None, None), doc='\n        Allows overriding the extents of the Element in 2D space defined\n        as four-tuple defining the (left, bottom, right and top) edges.')
    __abstract = True