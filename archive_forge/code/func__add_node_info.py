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
def _add_node_info(self, node_info):
    nodes = self.nodes.clone(datatype=['pandas', 'dictionary'])
    if isinstance(node_info, self.node_type):
        nodes = nodes.redim(**dict(zip(nodes.dimensions('key', label=True), node_info.kdims)))
    if not node_info.kdims and len(node_info) != len(nodes):
        raise ValueError('The supplied node data does not match the number of nodes defined by the edges. Ensure that the number of nodes matchor supply an index as the sole key dimension to allow the Graph to merge the data.')
    left_on = nodes.kdims[-1].name
    node_info_df = node_info.dframe()
    node_df = nodes.dframe()
    if node_info.kdims:
        idx = node_info.kdims[-1]
    else:
        idx = Dimension('index')
        node_info_df = node_info_df.reset_index()
    if 'index' in node_info_df.columns and (not idx.name == 'index'):
        node_df = node_df.rename(columns={'index': '__index'})
        left_on = '__index'
    cols = [c for c in node_info_df.columns if c not in node_df.columns or c == idx.name]
    node_info_df = node_info_df[cols]
    node_df = pd.merge(node_df, node_info_df, left_on=left_on, right_on=idx.name, how='left')
    nodes = nodes.clone(node_df, kdims=nodes.kdims[:2] + [idx], vdims=node_info.vdims)
    self._nodes = nodes