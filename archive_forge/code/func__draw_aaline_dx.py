from collections import namedtuple
from math import floor, ceil
def _draw_aaline_dx(d_x, slope, end, start, draw_two_pixel):
    g_x = ceil(start.x)
    g_y = start.y + (g_x - start.x) * slope
    if start.x < g_x:
        draw_two_pixel(floor(start.x), g_y - slope, inv_frac(start.x))
    rest = frac(end.x)
    s_x = ceil(end.x)
    if rest > 0:
        s_y = start.y + slope * (d_x + 1 - rest)
        draw_two_pixel(s_x, s_y, rest)
    else:
        s_x += 1
    for line_x in range(g_x, s_x):
        line_y = g_y + slope * (line_x - g_x)
        draw_two_pixel(line_x, line_y, 1)