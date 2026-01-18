from __future__ import division
import numpy as np
from pygsp import utils
@_plt_handle_figure
def _plt_plot_signal(G, signal, show_edges, limits, ax, vertex_size, highlight=[], colorbar=True):
    if show_edges:
        if G.is_directed():
            raise NotImplementedError
        elif G.coords.ndim == 1:
            pass
        elif G.coords.shape[1] == 2:
            x, y = _get_coords(G)
            ax.plot(x, y, linewidth=G.plotting['edge_width'], color=G.plotting['edge_color'], linestyle=G.plotting['edge_style'], zorder=1)
        elif G.coords.shape[1] == 3:
            x, y, z = _get_coords(G)
            for i in range(0, x.size, 2):
                x2, y2, z2 = (x[i:i + 2], y[i:i + 2], z[i:i + 2])
                ax.plot(x2, y2, z2, linewidth=G.plotting['edge_width'], color=G.plotting['edge_color'], linestyle=G.plotting['edge_style'], zorder=1)
    try:
        iter(highlight)
    except TypeError:
        highlight = [highlight]
    coords_hl = G.coords[highlight]
    if G.coords.ndim == 1:
        ax.plot(G.coords, signal)
        ax.set_ylim(limits)
        for coord_hl in coords_hl:
            ax.axvline(x=coord_hl, color='C1', linewidth=2)
    elif G.coords.shape[1] == 2:
        sc = ax.scatter(G.coords[:, 0], G.coords[:, 1], s=vertex_size, c=signal, zorder=2, vmin=limits[0], vmax=limits[1])
        ax.scatter(coords_hl[:, 0], coords_hl[:, 1], s=2 * vertex_size, zorder=3, marker='o', c='None', edgecolors='C1', linewidths=2)
    elif G.coords.shape[1] == 3:
        sc = ax.scatter(G.coords[:, 0], G.coords[:, 1], G.coords[:, 2], s=vertex_size, c=signal, zorder=2, vmin=limits[0], vmax=limits[1])
        ax.scatter(coords_hl[:, 0], coords_hl[:, 1], coords_hl[:, 2], s=2 * vertex_size, zorder=3, marker='o', c='None', edgecolors='C1', linewidths=2)
        try:
            ax.view_init(elev=G.plotting['elevation'], azim=G.plotting['azimuth'])
            ax.dist = G.plotting['distance']
        except KeyError:
            pass
    if G.coords.ndim != 1 and colorbar:
        plt = _import_plt()
        plt.colorbar(sc, ax=ax)