from __future__ import division
import numpy as np
from pygsp import utils
def _get_coords(G, edge_list=False):
    v_in, v_out, _ = G.get_edge_list()
    if edge_list:
        return np.stack((v_in, v_out), axis=1)
    coords = [np.stack((G.coords[v_in, d], G.coords[v_out, d]), axis=0) for d in range(G.coords.shape[1])]
    if G.coords.shape[1] == 2:
        return coords
    elif G.coords.shape[1] == 3:
        return [coord.reshape(-1, order='F') for coord in coords]