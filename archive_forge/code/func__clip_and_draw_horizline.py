from collections import namedtuple
from math import floor, ceil
def _clip_and_draw_horizline(surf, color, x_from, in_y, x_to):
    """draw clipped horizontal line."""
    clip = surf.get_clip()
    if in_y < clip.y or in_y >= clip.y + clip.h:
        return
    x_from = max(x_from, clip.x)
    x_to = min(x_to, clip.x + clip.w - 1)
    if x_to < clip.x or x_from >= clip.x + clip.w:
        return
    _drawhorzline(surf, color, x_from, in_y, x_to)