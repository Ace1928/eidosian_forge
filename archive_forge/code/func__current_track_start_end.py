from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def _current_track_start_end(self):
    track = self._parent[self.current_track_level]
    if track.start is None:
        start = self.start
    else:
        start = max(self.start, track.start)
    if track.end is None:
        end = self.end
    else:
        end = min(self.end, track.end)
    return (start, end)