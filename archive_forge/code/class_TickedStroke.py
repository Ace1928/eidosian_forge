from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
class TickedStroke(AbstractPathEffect):
    """
    A line-based PathEffect which draws a path with a ticked style.

    This line style is frequently used to represent constraints in
    optimization.  The ticks may be used to indicate that one side
    of the line is invalid or to represent a closed boundary of a
    domain (i.e. a wall or the edge of a pipe).

    The spacing, length, and angle of ticks can be controlled.

    This line style is sometimes referred to as a hatched line.

    See also the :doc:`/gallery/misc/tickedstroke_demo` example.
    """

    def __init__(self, offset=(0, 0), spacing=10.0, angle=45.0, length=np.sqrt(2), **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, in points.
        spacing : float, default: 10.0
            The spacing between ticks in points.
        angle : float, default: 45.0
            The angle between the path and the tick in degrees.  The angle
            is measured as if you were an ant walking along the curve, with
            zero degrees pointing directly ahead, 90 to your left, -90
            to your right, and 180 behind you. To change side of the ticks,
            change sign of the angle.
        length : float, default: 1.414
            The length of the tick relative to spacing.
            Recommended length = 1.414 (sqrt(2)) when angle=45, length=1.0
            when angle=90 and length=2.0 when angle=60.
        **kwargs
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.

        Examples
        --------
        See :doc:`/gallery/misc/tickedstroke_demo`.
        """
        super().__init__(offset)
        self._spacing = spacing
        self._angle = angle
        self._length = length
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """Draw the path with updated gc."""
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)
        gc0 = self._update_gc(gc0, self._gc)
        trans = affine + self._offset_transform(renderer)
        theta = -np.radians(self._angle)
        trans_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        spacing_px = renderer.points_to_pixels(self._spacing)
        transpath = affine.transform_path(tpath)
        polys = transpath.to_polygons(closed_only=False)
        for p in polys:
            x = p[:, 0]
            y = p[:, 1]
            if x.size < 2:
                continue
            ds = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])
            s = np.concatenate(([0.0], np.cumsum(ds)))
            s_total = s[-1]
            num = int(np.ceil(s_total / spacing_px)) - 1
            s_tick = np.linspace(spacing_px / 2, s_total - spacing_px / 2, num)
            x_tick = np.interp(s_tick, s, x)
            y_tick = np.interp(s_tick, s, y)
            delta_s = self._spacing * 0.001
            u = (np.interp(s_tick + delta_s, s, x) - x_tick) / delta_s
            v = (np.interp(s_tick + delta_s, s, y) - y_tick) / delta_s
            n = np.hypot(u, v)
            mask = n == 0
            n[mask] = 1.0
            uv = np.array([u / n, v / n]).T
            uv[mask] = np.array([0, 0]).T
            dxy = np.dot(uv, trans_matrix) * self._length * spacing_px
            x_end = x_tick + dxy[:, 0]
            y_end = y_tick + dxy[:, 1]
            xyt = np.empty((2 * num, 2), dtype=x_tick.dtype)
            xyt[0::2, 0] = x_tick
            xyt[1::2, 0] = x_end
            xyt[0::2, 1] = y_tick
            xyt[1::2, 1] = y_end
            codes = np.tile([Path.MOVETO, Path.LINETO], num)
            h = Path(xyt, codes)
            renderer.draw_path(gc0, h, affine.inverted() + trans, rgbFace)
        gc0.restore()