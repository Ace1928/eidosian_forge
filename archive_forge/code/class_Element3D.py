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
class Element3D(Element2D):
    extents = param.Tuple(default=(None, None, None, None, None, None), doc='\n        Allows overriding the extents of the Element in 3D space\n        defined as (xmin, ymin, zmin, xmax, ymax, zmax).')
    __abstract = True
    _selection_streams = ()