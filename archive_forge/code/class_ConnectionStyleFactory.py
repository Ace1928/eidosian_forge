import collections
import itertools
from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
class ConnectionStyleFactory:

    def __init__(self, connectionstyles, selfloop_height, ax=None):
        import matplotlib as mpl
        import matplotlib.path
        import numpy as np
        self.ax = ax
        self.mpl = mpl
        self.np = np
        self.base_connection_styles = [mpl.patches.ConnectionStyle(cs) for cs in connectionstyles]
        self.n = len(self.base_connection_styles)
        self.selfloop_height = selfloop_height

    def curved(self, edge_index):
        return self.base_connection_styles[edge_index % self.n]

    def self_loop(self, edge_index):

        def self_loop_connection(posA, posB, *args, **kwargs):
            if not self.np.all(posA == posB):
                raise nx.NetworkXError('`self_loop` connection style methodis only to be used for self-loops')
            data_loc = self.ax.transData.inverted().transform(posA)
            v_shift = 0.1 * self.selfloop_height
            h_shift = v_shift * 0.5
            path = self.np.asarray([[0, v_shift], [h_shift, v_shift], [h_shift, 0], [0, 0], [-h_shift, 0], [-h_shift, v_shift], [0, v_shift]])
            if edge_index % 4:
                x, y = path.T
                for _ in range(edge_index % 4):
                    x, y = (y, -x)
                path = self.np.array([x, y]).T
            return self.mpl.path.Path(self.ax.transData.transform(data_loc + path), [1, 4, 4, 4, 4, 4, 4])
        return self_loop_connection