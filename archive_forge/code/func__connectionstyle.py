from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def _connectionstyle(posA, posB, *args, **kwargs):
    if np.all(posA == posB):
        selfloop_ht = 0.005 * max_nodesize if h == 0 else h
        data_loc = ax.transData.inverted().transform(posA)
        v_shift = 0.1 * selfloop_ht
        h_shift = v_shift * 0.5
        path = [data_loc + np.asarray([0, v_shift]), data_loc + np.asarray([h_shift, v_shift]), data_loc + np.asarray([h_shift, 0]), data_loc, data_loc + np.asarray([-h_shift, 0]), data_loc + np.asarray([-h_shift, v_shift]), data_loc + np.asarray([0, v_shift])]
        ret = mpl.path.Path(ax.transData.transform(path), [1, 4, 4, 4, 4, 4, 4])
    else:
        ret = base_connection_style(posA, posB, *args, **kwargs)
    return ret