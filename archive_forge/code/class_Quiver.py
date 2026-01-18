import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
class Quiver(mcollections.PolyCollection):
    """
    Specialized PolyCollection for arrows.

    The only API method is set_UVC(), which can be used
    to change the size, orientation, and color of the
    arrows; their locations are fixed when the class is
    instantiated.  Possibly this method will be useful
    in animations.

    Much of the work in this class is done in the draw()
    method so that as much information as possible is available
    about the plot.  In subsequent draw() calls, recalculation
    is limited to things that might have changed, so there
    should be no performance penalty from putting the calculations
    in the draw() method.
    """
    _PIVOT_VALS = ('tail', 'middle', 'tip')

    @_docstring.Substitution(_quiver_doc)
    def __init__(self, ax, *args, scale=None, headwidth=3, headlength=5, headaxislength=4.5, minshaft=1, minlength=1, units='width', scale_units=None, angles='uv', width=None, color='k', pivot='tail', **kwargs):
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pyplot interface documentation:
        %s
        """
        self._axes = ax
        X, Y, U, V, C = _parse_args(*args, caller_name='quiver')
        self.X = X
        self.Y = Y
        self.XY = np.column_stack((X, Y))
        self.N = len(X)
        self.scale = scale
        self.headwidth = headwidth
        self.headlength = float(headlength)
        self.headaxislength = headaxislength
        self.minshaft = minshaft
        self.minlength = minlength
        self.units = units
        self.scale_units = scale_units
        self.angles = angles
        self.width = width
        if pivot.lower() == 'mid':
            pivot = 'middle'
        self.pivot = pivot.lower()
        _api.check_in_list(self._PIVOT_VALS, pivot=self.pivot)
        self.transform = kwargs.pop('transform', ax.transData)
        kwargs.setdefault('facecolors', color)
        kwargs.setdefault('linewidths', (0,))
        super().__init__([], offsets=self.XY, offset_transform=self.transform, closed=False, **kwargs)
        self.polykw = kwargs
        self.set_UVC(U, V, C)
        self._dpi_at_last_init = None

    def _init(self):
        """
        Initialization delayed until first draw;
        allow time for axes setup.
        """
        if True:
            trans = self._set_transform()
            self.span = trans.inverted().transform_bbox(self.axes.bbox).width
            if self.width is None:
                sn = np.clip(math.sqrt(self.N), 8, 25)
                self.width = 0.06 * self.span / sn
            if self._dpi_at_last_init != self.axes.figure.dpi and self.scale is None:
                self._make_verts(self.XY, self.U, self.V, self.angles)
            self._dpi_at_last_init = self.axes.figure.dpi

    def get_datalim(self, transData):
        trans = self.get_transform()
        offset_trf = self.get_offset_transform()
        full_transform = trans - transData + (offset_trf - transData)
        XY = full_transform.transform(self.XY)
        bbox = transforms.Bbox.null()
        bbox.update_from_data_xy(XY, ignore=True)
        return bbox

    @martist.allow_rasterization
    def draw(self, renderer):
        self._init()
        verts = self._make_verts(self.XY, self.U, self.V, self.angles)
        self.set_verts(verts, closed=False)
        super().draw(renderer)
        self.stale = False

    def set_UVC(self, U, V, C=None):
        U = ma.masked_invalid(U, copy=True).ravel()
        V = ma.masked_invalid(V, copy=True).ravel()
        if C is not None:
            C = ma.masked_invalid(C, copy=True).ravel()
        for name, var in zip(('U', 'V', 'C'), (U, V, C)):
            if not (var is None or var.size == self.N or var.size == 1):
                raise ValueError(f'Argument {name} has a size {var.size} which does not match {self.N}, the number of arrow positions')
        mask = ma.mask_or(U.mask, V.mask, copy=False, shrink=True)
        if C is not None:
            mask = ma.mask_or(mask, C.mask, copy=False, shrink=True)
            if mask is ma.nomask:
                C = C.filled()
            else:
                C = ma.array(C, mask=mask, copy=False)
        self.U = U.filled(1)
        self.V = V.filled(1)
        self.Umask = mask
        if C is not None:
            self.set_array(C)
        self.stale = True

    def _dots_per_unit(self, units):
        """Return a scale factor for converting from units to pixels."""
        bb = self.axes.bbox
        vl = self.axes.viewLim
        return _api.check_getitem({'x': bb.width / vl.width, 'y': bb.height / vl.height, 'xy': np.hypot(*bb.size) / np.hypot(*vl.size), 'width': bb.width, 'height': bb.height, 'dots': 1.0, 'inches': self.axes.figure.dpi}, units=units)

    def _set_transform(self):
        """
        Set the PolyCollection transform to go
        from arrow width units to pixels.
        """
        dx = self._dots_per_unit(self.units)
        self._trans_scale = dx
        trans = transforms.Affine2D().scale(dx)
        self.set_transform(trans)
        return trans

    def _angles_lengths(self, XY, U, V, eps=1):
        xy = self.axes.transData.transform(XY)
        uv = np.column_stack((U, V))
        xyp = self.axes.transData.transform(XY + eps * uv)
        dxy = xyp - xy
        angles = np.arctan2(dxy[:, 1], dxy[:, 0])
        lengths = np.hypot(*dxy.T) / eps
        return (angles, lengths)

    def _make_verts(self, XY, U, V, angles):
        uv = U + V * 1j
        str_angles = angles if isinstance(angles, str) else ''
        if str_angles == 'xy' and self.scale_units == 'xy':
            angles, lengths = self._angles_lengths(XY, U, V, eps=1)
        elif str_angles == 'xy' or self.scale_units == 'xy':
            eps = np.abs(self.axes.dataLim.extents).max() * 0.001
            angles, lengths = self._angles_lengths(XY, U, V, eps=eps)
        if str_angles and self.scale_units == 'xy':
            a = lengths
        else:
            a = np.abs(uv)
        if self.scale is None:
            sn = max(10, math.sqrt(self.N))
            if self.Umask is not ma.nomask:
                amean = a[~self.Umask].mean()
            else:
                amean = a.mean()
            scale = 1.8 * amean * sn / self.span
        if self.scale_units is None:
            if self.scale is None:
                self.scale = scale
            widthu_per_lenu = 1.0
        else:
            if self.scale_units == 'xy':
                dx = 1
            else:
                dx = self._dots_per_unit(self.scale_units)
            widthu_per_lenu = dx / self._trans_scale
            if self.scale is None:
                self.scale = scale * widthu_per_lenu
        length = a * (widthu_per_lenu / (self.scale * self.width))
        X, Y = self._h_arrows(length)
        if str_angles == 'xy':
            theta = angles
        elif str_angles == 'uv':
            theta = np.angle(uv)
        else:
            theta = ma.masked_invalid(np.deg2rad(angles)).filled(0)
        theta = theta.reshape((-1, 1))
        xy = (X + Y * 1j) * np.exp(1j * theta) * self.width
        XY = np.stack((xy.real, xy.imag), axis=2)
        if self.Umask is not ma.nomask:
            XY = ma.array(XY)
            XY[self.Umask] = ma.masked
        return XY

    def _h_arrows(self, length):
        """Length is in arrow width units."""
        minsh = self.minshaft * self.headlength
        N = len(length)
        length = length.reshape(N, 1)
        np.clip(length, 0, 2 ** 16, out=length)
        x = np.array([0, -self.headaxislength, -self.headlength, 0], np.float64)
        x = x + np.array([0, 1, 1, 1]) * length
        y = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        y = np.repeat(y[np.newaxis, :], N, axis=0)
        x0 = np.array([0, minsh - self.headaxislength, minsh - self.headlength, minsh], np.float64)
        y0 = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        ii = [0, 1, 2, 3, 2, 1, 0, 0]
        X = x[:, ii]
        Y = y[:, ii]
        Y[:, 3:-1] *= -1
        X0 = x0[ii]
        Y0 = y0[ii]
        Y0[3:-1] *= -1
        shrink = length / minsh if minsh != 0.0 else 0.0
        X0 = shrink * X0[np.newaxis, :]
        Y0 = shrink * Y0[np.newaxis, :]
        short = np.repeat(length < minsh, 8, axis=1)
        np.copyto(X, X0, where=short)
        np.copyto(Y, Y0, where=short)
        if self.pivot == 'middle':
            X -= 0.5 * X[:, 3, np.newaxis]
        elif self.pivot == 'tip':
            X = X - X[:, 3, np.newaxis]
        elif self.pivot != 'tail':
            _api.check_in_list(['middle', 'tip', 'tail'], pivot=self.pivot)
        tooshort = length < self.minlength
        if tooshort.any():
            th = np.arange(0, 8, 1, np.float64) * (np.pi / 3.0)
            x1 = np.cos(th) * self.minlength * 0.5
            y1 = np.sin(th) * self.minlength * 0.5
            X1 = np.repeat(x1[np.newaxis, :], N, axis=0)
            Y1 = np.repeat(y1[np.newaxis, :], N, axis=0)
            tooshort = np.repeat(tooshort, 8, 1)
            np.copyto(X, X1, where=tooshort)
            np.copyto(Y, Y1, where=tooshort)
        return (X, Y)
    quiver_doc = _api.deprecated('3.7')(property(lambda self: _quiver_doc))