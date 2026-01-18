from collections import namedtuple
from math import floor, ceil
def _draw_polygon_inner_loop(index, point_x, point_y, y_coord, x_intersect):
    i_prev = index - 1 if index else len(point_x) - 1
    y_1 = point_y[i_prev]
    y_2 = point_y[index]
    if y_1 < y_2:
        x_1 = point_x[i_prev]
        x_2 = point_x[index]
    elif y_1 > y_2:
        y_2 = point_y[i_prev]
        y_1 = point_y[index]
        x_2 = point_x[i_prev]
        x_1 = point_x[index]
    else:
        return
    if y_2 > y_coord >= y_1 or (y_coord == max(point_y) and y_coord <= y_2):
        x_intersect.append((y_coord - y_1) * (x_2 - x_1) // (y_2 - y_1) + x_1)