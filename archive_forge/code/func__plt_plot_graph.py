from __future__ import division
import numpy as np
from pygsp import utils
@_plt_handle_figure
def _plt_plot_graph(G, show_edges, vertex_size, ax):
    if show_edges:
        if G.is_directed():
            raise NotImplementedError
        else:
            if G.coords.shape[1] == 2:
                x, y = _get_coords(G)
                ax.plot(x, y, linewidth=G.plotting['edge_width'], color=G.plotting['edge_color'], linestyle=G.plotting['edge_style'], marker='o', markersize=vertex_size / 10, markerfacecolor=G.plotting['vertex_color'], markeredgecolor=G.plotting['vertex_color'])
            if G.coords.shape[1] == 3:
                x, y, z = _get_coords(G)
                for i in range(0, x.size, 2):
                    x2, y2, z2 = (x[i:i + 2], y[i:i + 2], z[i:i + 2])
                    ax.plot(x2, y2, z2, linewidth=G.plotting['edge_width'], color=G.plotting['edge_color'], linestyle=G.plotting['edge_style'], marker='o', markersize=vertex_size / 10, markerfacecolor=G.plotting['vertex_color'], markeredgecolor=G.plotting['vertex_color'])
    else:
        if G.coords.shape[1] == 2:
            ax.scatter(G.coords[:, 0], G.coords[:, 1], marker='o', s=vertex_size, c=G.plotting['vertex_color'])
        if G.coords.shape[1] == 3:
            ax.scatter(G.coords[:, 0], G.coords[:, 1], G.coords[:, 2], marker='o', s=vertex_size, c=G.plotting['vertex_color'])
    if G.coords.shape[1] == 3:
        try:
            ax.view_init(elev=G.plotting['elevation'], azim=G.plotting['azimuth'])
            ax.dist = G.plotting['distance']
        except KeyError:
            pass