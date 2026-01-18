import uuid
import warnings
from ast import literal_eval
from collections import Counter, defaultdict
from functools import partial
from itertools import groupby, product
import numpy as np
import param
from panel.config import config
from panel.io.document import unlocked
from panel.io.notebook import push
from panel.io.state import state
from pyviz_comms import JupyterComm
from ..core import traversal, util
from ..core.data import Dataset, disable_pipeline
from ..core.element import Element, Element3D
from ..core.layout import Empty, Layout, NdLayout
from ..core.options import Compositor, SkipRendering, Store, lookup_options
from ..core.overlay import CompositeOverlay, NdOverlay, Overlay
from ..core.spaces import DynamicMap, HoloMap
from ..core.util import isfinite, stream_parameters
from ..element import Graph, Table
from ..selection import NoOpSelectionDisplay
from ..streams import RangeX, RangeXY, RangeY, Stream
from ..util.transform import dim
from .util import (
@classmethod
def _merge_group_ranges(cls, ranges):
    hard_range = util.max_range(ranges['hard'], combined=False)
    soft_range = util.max_range(ranges['soft'])
    robust_range = util.max_range(ranges.get('robust', []))
    data_range = util.max_range(ranges['data'])
    combined = util.dimension_range(data_range[0], data_range[1], hard_range, soft_range)
    dranges = {'data': data_range, 'hard': hard_range, 'soft': soft_range, 'combined': combined, 'robust': robust_range, 'values': ranges}
    if 'factors' in ranges:
        all_factors = ranges['factors']
        factor_dtypes = {fs.dtype for fs in all_factors} if all_factors else []
        dtype = next(iter(factor_dtypes)) if len(factor_dtypes) == 1 else None
        expanded = [v for fctrs in all_factors for v in fctrs]
        if dtype is not None:
            try:
                expanded = np.array(expanded, dtype=dtype)
            except Exception:
                pass
        dranges['factors'] = util.unique_array(expanded)
    return dranges