from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
def _build_curves(self):
    self.curves = curves = []
    self.polygons = []
    for polyline, color in self.polylines:
        n = len(curves)
        polygon = []
        for arc in polyline:
            polygon += arc[1:-1]
            if arc[0] == arc[-1]:
                A = SmoothLoop(self.canvas, arc, color, tension1=self.tension1, tension2=self.tension2)
                curves.append(A)
            else:
                A = SmoothArc(self.canvas, arc, color, tension1=self.tension1, tension2=self.tension2)
                curves.append(A)
        self.polygons.append(polygon)