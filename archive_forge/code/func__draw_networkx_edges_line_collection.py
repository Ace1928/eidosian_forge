from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def _draw_networkx_edges_line_collection():
    edge_collection = mpl.collections.LineCollection(edge_pos, colors=edge_color, linewidths=width, antialiaseds=(1,), linestyle=style, alpha=alpha)
    edge_collection.set_cmap(edge_cmap)
    edge_collection.set_clim(edge_vmin, edge_vmax)
    edge_collection.set_zorder(1)
    edge_collection.set_label(label)
    ax.add_collection(edge_collection)
    return edge_collection