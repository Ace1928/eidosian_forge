import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
class QuiverKey(martist.Artist):
    """Labelled arrow for use as a quiver plot scale key."""
    halign = {'N': 'center', 'S': 'center', 'E': 'left', 'W': 'right'}
    valign = {'N': 'bottom', 'S': 'top', 'E': 'center', 'W': 'center'}
    pivot = {'N': 'middle', 'S': 'middle', 'E': 'tip', 'W': 'tail'}

    def __init__(self, Q, X, Y, U, label, *, angle=0, coordinates='axes', color=None, labelsep=0.1, labelpos='N', labelcolor=None, fontproperties=None, **kwargs):
        """
        Add a key to a quiver plot.

        The positioning of the key depends on *X*, *Y*, *coordinates*, and
        *labelpos*.  If *labelpos* is 'N' or 'S', *X*, *Y* give the position of
        the middle of the key arrow.  If *labelpos* is 'E', *X*, *Y* positions
        the head, and if *labelpos* is 'W', *X*, *Y* positions the tail; in
        either of these two cases, *X*, *Y* is somewhere in the middle of the
        arrow+label key object.

        Parameters
        ----------
        Q : `~matplotlib.quiver.Quiver`
            A `.Quiver` object as returned by a call to `~.Axes.quiver()`.
        X, Y : float
            The location of the key.
        U : float
            The length of the key.
        label : str
            The key label (e.g., length and units of the key).
        angle : float, default: 0
            The angle of the key arrow, in degrees anti-clockwise from the
            horizontal axis.
        coordinates : {'axes', 'figure', 'data', 'inches'}, default: 'axes'
            Coordinate system and units for *X*, *Y*: 'axes' and 'figure' are
            normalized coordinate systems with (0, 0) in the lower left and
            (1, 1) in the upper right; 'data' are the axes data coordinates
            (used for the locations of the vectors in the quiver plot itself);
            'inches' is position in the figure in inches, with (0, 0) at the
            lower left corner.
        color : color
            Overrides face and edge colors from *Q*.
        labelpos : {'N', 'S', 'E', 'W'}
            Position the label above, below, to the right, to the left of the
            arrow, respectively.
        labelsep : float, default: 0.1
            Distance in inches between the arrow and the label.
        labelcolor : color, default: :rc:`text.color`
            Label color.
        fontproperties : dict, optional
            A dictionary with keyword arguments accepted by the
            `~matplotlib.font_manager.FontProperties` initializer:
            *family*, *style*, *variant*, *size*, *weight*.
        **kwargs
            Any additional keyword arguments are used to override vector
            properties taken from *Q*.
        """
        super().__init__()
        self.Q = Q
        self.X = X
        self.Y = Y
        self.U = U
        self.angle = angle
        self.coord = coordinates
        self.color = color
        self.label = label
        self._labelsep_inches = labelsep
        self.labelpos = labelpos
        self.labelcolor = labelcolor
        self.fontproperties = fontproperties or dict()
        self.kw = kwargs
        self.text = mtext.Text(text=label, horizontalalignment=self.halign[self.labelpos], verticalalignment=self.valign[self.labelpos], fontproperties=self.fontproperties)
        if self.labelcolor is not None:
            self.text.set_color(self.labelcolor)
        self._dpi_at_last_init = None
        self.zorder = Q.zorder + 0.1

    @property
    def labelsep(self):
        return self._labelsep_inches * self.Q.axes.figure.dpi

    def _init(self):
        if True:
            if self.Q._dpi_at_last_init != self.Q.axes.figure.dpi:
                self.Q._init()
            self._set_transform()
            with cbook._setattr_cm(self.Q, pivot=self.pivot[self.labelpos], Umask=ma.nomask):
                u = self.U * np.cos(np.radians(self.angle))
                v = self.U * np.sin(np.radians(self.angle))
                self.verts = self.Q._make_verts([[0.0, 0.0]], np.array([u]), np.array([v]), 'uv')
            kwargs = self.Q.polykw
            kwargs.update(self.kw)
            self.vector = mcollections.PolyCollection(self.verts, offsets=[(self.X, self.Y)], offset_transform=self.get_transform(), **kwargs)
            if self.color is not None:
                self.vector.set_color(self.color)
            self.vector.set_transform(self.Q.get_transform())
            self.vector.set_figure(self.get_figure())
            self._dpi_at_last_init = self.Q.axes.figure.dpi

    def _text_shift(self):
        return {'N': (0, +self.labelsep), 'S': (0, -self.labelsep), 'E': (+self.labelsep, 0), 'W': (-self.labelsep, 0)}[self.labelpos]

    @martist.allow_rasterization
    def draw(self, renderer):
        self._init()
        self.vector.draw(renderer)
        pos = self.get_transform().transform((self.X, self.Y))
        self.text.set_position(pos + self._text_shift())
        self.text.draw(renderer)
        self.stale = False

    def _set_transform(self):
        self.set_transform(_api.check_getitem({'data': self.Q.axes.transData, 'axes': self.Q.axes.transAxes, 'figure': self.Q.axes.figure.transFigure, 'inches': self.Q.axes.figure.dpi_scale_trans}, coordinates=self.coord))

    def set_figure(self, fig):
        super().set_figure(fig)
        self.text.set_figure(fig)

    def contains(self, mouseevent):
        if self._different_canvas(mouseevent):
            return (False, {})
        if self.text.contains(mouseevent)[0] or self.vector.contains(mouseevent)[0]:
            return (True, {})
        return (False, {})