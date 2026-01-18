from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
class SmoothLoop(SmoothArc):
    """
    A Bezier spline that is tangent at the midpoints of segments in a
    PL loop given by specifying a list of vertices.  Speeds at
    the spline knots are chosen by using Hobby's scheme.
    """

    def __init__(self, canvas, vertices, color='black', tension1=1.0, tension2=1.0):
        self.canvas = canvas
        if vertices[0] != vertices[-1]:
            vertices.append(vertices[0])
        vertices.append(vertices[1])
        self.vertices = V = [TwoVector(*p) for p in vertices]
        self.tension1, self.tension2 = (tension1, tension2)
        self.color = color
        self.canvas_items = []
        self.spline_knots = [0.5 * (V[k] + V[k + 1]) for k in range(len(V) - 1)]
        self.spline_knots.append(self.spline_knots[0])
        self.tangents = [V[k + 1] - V[k] for k in range(len(V) - 1)]
        self.tangents.append(self.tangents[0])
        assert len(self.spline_knots) == len(self.tangents)