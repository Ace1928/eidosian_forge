from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def _draw_networkx_edges_fancy_arrow_patch():

    def to_marker_edge(marker_size, marker):
        if marker in 's^>v<d':
            return np.sqrt(2 * marker_size) / 2
        else:
            return np.sqrt(marker_size) / 2
    arrow_collection = []
    if isinstance(arrowsize, list):
        if len(arrowsize) != len(edge_pos):
            raise ValueError('arrowsize should have the same length as edgelist')
    else:
        mutation_scale = arrowsize
    base_connection_style = mpl.patches.ConnectionStyle(connectionstyle)
    max_nodesize = np.array(node_size).max()

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
    arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
    for i, (src, dst) in zip(fancy_edges_indices, edge_pos):
        x1, y1 = src
        x2, y2 = dst
        shrink_source = 0
        shrink_target = 0
        if isinstance(arrowsize, list):
            mutation_scale = arrowsize[i]
        if np.iterable(node_size):
            source, target = edgelist[i][:2]
            source_node_size = node_size[nodelist.index(source)]
            target_node_size = node_size[nodelist.index(target)]
            shrink_source = to_marker_edge(source_node_size, node_shape)
            shrink_target = to_marker_edge(target_node_size, node_shape)
        else:
            shrink_source = shrink_target = to_marker_edge(node_size, node_shape)
        if shrink_source < min_source_margin:
            shrink_source = min_source_margin
        if shrink_target < min_target_margin:
            shrink_target = min_target_margin
        if len(arrow_colors) > i:
            arrow_color = arrow_colors[i]
        elif len(arrow_colors) == 1:
            arrow_color = arrow_colors[0]
        else:
            arrow_color = arrow_colors[i % len(arrow_colors)]
        if np.iterable(width):
            if len(width) > i:
                line_width = width[i]
            else:
                line_width = width[i % len(width)]
        else:
            line_width = width
        if np.iterable(style) and (not isinstance(style, str)) and (not isinstance(style, tuple)):
            if len(style) > i:
                linestyle = style[i]
            else:
                linestyle = style[i % len(style)]
        else:
            linestyle = style
        arrow = mpl.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=arrowstyle, shrinkA=shrink_source, shrinkB=shrink_target, mutation_scale=mutation_scale, color=arrow_color, linewidth=line_width, connectionstyle=_connectionstyle, linestyle=linestyle, zorder=1)
        arrow_collection.append(arrow)
        ax.add_patch(arrow)
    return arrow_collection