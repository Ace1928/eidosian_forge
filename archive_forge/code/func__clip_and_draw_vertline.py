from collections import namedtuple
from math import floor, ceil
def _clip_and_draw_vertline(surf, color, in_x, y_from, y_to):
    """draw clipped vertical line."""
    clip = surf.get_clip()
    if in_x < clip.x or in_x >= clip.x + clip.w:
        return
    y_from = max(y_from, clip.y)
    y_to = min(y_to, clip.y + clip.h - 1)
    if y_to < clip.y or y_from >= clip.y + clip.h:
        return
    _drawvertline(surf, color, in_x, y_from, y_to)