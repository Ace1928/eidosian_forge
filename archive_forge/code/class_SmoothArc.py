from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
class SmoothArc:
    """
    A Bezier spline that is tangent at the midpoints of segments in
    the PL path given by specifying a list of vertices.  Speeds
    at the spline knots are chosen by using Hobby's scheme.
    """

    def __init__(self, canvas, vertices, color='black', tension1=1.0, tension2=1.0):
        self.canvas = canvas
        self.vertices = V = [TwoVector(*p) for p in vertices]
        self.tension1, self.tension2 = (tension1, tension2)
        self.color = color
        self.canvas_items = []
        self.spline_knots = K = [V[0]] + [0.5 * (V[k] + V[k + 1]) for k in range(1, len(V) - 2)] + [V[-1]]
        self.tangents = [V[1] - K[0]] + [V[k + 1] - K[k] for k in range(1, len(V) - 2)] + [V[-1] - V[-2]]
        assert len(self.spline_knots) == len(self.tangents)

    def _polar_to_vector(self, r, phi):
        """
        Return a TwoVector with specified length and angle.
        """
        return TwoVector(r * cos(phi), r * sin(phi))

    def _curve_to(self, k):
        """
        Compute the two control points for a nice cubic curve from the
        kth spline knot to the next one.  Return the kth spline knot
        and the two control points.  We do not allow the speed at the
        spline knots to exceed the distance to the interlacing vertex
        of the PL curve; this avoids extraneous inflection points.
        """
        A, B = self.spline_knots[k:k + 2]
        vA, vB = self.tangents[k:k + 2]
        A_speed_max, B_speed_max = (abs(vA), abs(vB))
        base = B - A
        l, psi = (abs(base), base.angle())
        theta, phi = (vA.angle() - psi, psi - vB.angle())
        ctheta, stheta = (cos(theta), sin(theta))
        cphi, sphi = (cos(phi), sin(phi))
        a = sqrt(2.0)
        b = 1.0 / 16.0
        c = (3.0 - sqrt(5.0)) / 2.0
        alpha = a * (stheta - b * sphi) * (sphi - b * stheta) * (ctheta - cphi)
        rho = (2 + alpha) / ((1 + (1 - c) * ctheta + c * cphi) * self.tension1)
        sigma = (2 - alpha) / ((1 + (1 - c) * cphi + c * ctheta) * self.tension2)
        A_speed = min(l * rho / 3, A_speed_max)
        B_speed = min(l * sigma / 3, B_speed_max)
        return [A, A + self._polar_to_vector(A_speed, psi + theta), B - self._polar_to_vector(B_speed, psi - phi)]

    def bezier(self):
        """
        Return a list of spline knots and control points for the Bezier
        spline, in format [ ... Knot, Control, Control, Knot ...]
        """
        path = []
        for k in range(len(self.spline_knots) - 1):
            path += self._curve_to(k)
        path.append(self.spline_knots[-1])
        return path

    def tk_clear(self):
        for item in self.canvas_items:
            self.canvas.delete(item)

    def tk_draw(self, thickness=4):
        XY = self.bezier()
        self.tk_clear()
        self.canvas_items.append(self.canvas.create_line(*XY, smooth='raw', width=thickness, fill=self.color, capstyle=Tk_.ROUND, splinesteps=100, tags=('smooth', 'transformable')))

    def pyx_draw(self, canvas, transform):
        XY = [transform(xy) for xy in self.bezier()]
        arc_parts = [pyx.path.moveto(*XY[0])]
        for i in range(1, len(XY), 3):
            arc_parts.append(pyx.path.curveto(XY[i][0], XY[i][1], XY[i + 1][0], XY[i + 1][1], XY[i + 2][0], XY[i + 2][1]))
            style = [pyx.style.linewidth(4), pyx.style.linecap.round, pyx.color.rgbfromhexstring(self.color)]
            path = pyx.path.path(*arc_parts)
            canvas.stroke(path, style)

    def tikz_draw(self, file, transform):
        points = ['(%.2f, %.2f)' % transform(xy) for xy in self.bezier()]
        file.write(self.color, '    \\draw %s .. controls %s and %s .. ' % tuple(points[:3]))
        for i in range(3, len(points) - 3, 3):
            file.write(self.color, '\n' + 10 * ' ' + '%s .. controls %s and %s .. ' % tuple(points[i:i + 3]))
        file.write(self.color, points[-1] + ';\n')