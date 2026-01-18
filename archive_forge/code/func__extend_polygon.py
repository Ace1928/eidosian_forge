from functools import reduce
import numpy as np
from ..draw import polygon
from .._shared.version_requirements import require
def _extend_polygon(event):
    if event.inaxes is None or event.inaxes is undo_pos:
        return
    if ax.get_navigate_mode():
        return
    if event.button == LEFT_CLICK:
        temp_list.append([event.xdata, event.ydata])
        if preview_polygon_drawn:
            poly = preview_polygon_drawn.pop()
            poly.remove()
        polygon = _draw_polygon(ax, temp_list, alpha=alpha / 1.4)
        preview_polygon_drawn.append(polygon)
    elif event.button == RIGHT_CLICK:
        if not temp_list:
            return
        list_of_vertex_lists.append(temp_list[:])
        polygon_object = _draw_polygon(ax, temp_list, alpha=alpha)
        polygons_drawn.append(polygon_object)
        preview_poly = preview_polygon_drawn.pop()
        preview_poly.remove()
        del temp_list[:]
        plt.draw()