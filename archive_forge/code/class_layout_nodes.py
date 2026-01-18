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
class layout_nodes(Operation):
    """
    Accepts a Graph and lays out the corresponding nodes with the
    supplied networkx layout function. If no layout function is
    supplied uses a simple circular_layout function. Also supports
    LayoutAlgorithm function provided in datashader layouts.
    """
    only_nodes = param.Boolean(default=False, doc='\n        Whether to return Nodes or Graph.')
    layout = param.Callable(default=None, doc='\n        A NetworkX layout function')
    kwargs = param.Dict(default={}, doc='\n        Keyword arguments passed to the layout function.')

    def _process(self, element, key=None):
        if self.p.layout and isinstance(self.p.layout, FunctionType):
            import networkx as nx
            edges = element.array([0, 1])
            graph = nx.from_edgelist(edges)
            if 'weight' in self.p.kwargs:
                weight = self.p.kwargs['weight']
                for (s, t), w in zip(edges, element[weight]):
                    graph.edges[s, t][weight] = w
            positions = self.p.layout(graph, **self.p.kwargs)
            nodes = [tuple(pos) + (idx,) for idx, pos in sorted(positions.items())]
        else:
            source = element.dimension_values(0, expanded=False)
            target = element.dimension_values(1, expanded=False)
            nodes = np.unique(np.concatenate([source, target]))
            if self.p.layout:
                df = pd.DataFrame({'index': nodes})
                nodes = self.p.layout(df, element.dframe(), **self.p.kwargs)
                nodes = nodes[['x', 'y', 'index']]
            else:
                nodes = circular_layout(nodes)
        nodes = element.node_type(nodes)
        if element._nodes:
            for d in element.nodes.vdims:
                vals = element.nodes.dimension_values(d)
                nodes = nodes.add_dimension(d, len(nodes.vdims), vals, vdim=True)
        if self.p.only_nodes:
            return nodes
        return element.clone((element.data, nodes))