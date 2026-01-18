import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
class PcolorImage(AxesImage):
    """
    Make a pcolor-style plot with an irregular rectangular grid.

    This uses a variation of the original irregular image code,
    and it is used by pcolorfast for the corresponding grid type.
    """

    def __init__(self, ax, x=None, y=None, A=None, *, cmap=None, norm=None, **kwargs):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes the image will belong to.
        x, y : 1D array-like, optional
            Monotonic arrays of length N+1 and M+1, respectively, specifying
            rectangle boundaries.  If not given, will default to
            ``range(N + 1)`` and ``range(M + 1)``, respectively.
        A : array-like
            The data to be color-coded. The interpretation depends on the
            shape:

            - (M, N) `~numpy.ndarray` or masked array: values to be colormapped
            - (M, N, 3): RGB array
            - (M, N, 4): RGBA array

        cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            The Colormap instance or registered colormap name used to map
            scalar data to colors.
        norm : str or `~matplotlib.colors.Normalize`
            Maps luminance to 0-1.
        **kwargs : `~matplotlib.artist.Artist` properties
        """
        super().__init__(ax, norm=norm, cmap=cmap)
        self._internal_update(kwargs)
        if A is not None:
            self.set_data(x, y, A)

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        if self._A is None:
            raise RuntimeError('You must first set the image array')
        if unsampled:
            raise ValueError('unsampled not supported on PColorImage')
        if self._imcache is None:
            A = self.to_rgba(self._A, bytes=True)
            self._imcache = np.pad(A, [(1, 1), (1, 1), (0, 0)], 'constant')
        padded_A = self._imcache
        bg = mcolors.to_rgba(self.axes.patch.get_facecolor(), 0)
        bg = (np.array(bg) * 255).astype(np.uint8)
        if (padded_A[0, 0] != bg).all():
            padded_A[[0, -1], :] = padded_A[:, [0, -1]] = bg
        l, b, r, t = self.axes.bbox.extents
        width = round(r) + 0.5 - (round(l) - 0.5)
        height = round(t) + 0.5 - (round(b) - 0.5)
        width = round(width * magnification)
        height = round(height * magnification)
        vl = self.axes.viewLim
        x_pix = np.linspace(vl.x0, vl.x1, width)
        y_pix = np.linspace(vl.y0, vl.y1, height)
        x_int = self._Ax.searchsorted(x_pix)
        y_int = self._Ay.searchsorted(y_pix)
        im = padded_A.view(np.uint32).ravel()[np.add.outer(y_int * padded_A.shape[1], x_int)].view(np.uint8).reshape((height, width, 4))
        return (im, l, b, IdentityTransform())

    def _check_unsampled_image(self):
        return False

    def set_data(self, x, y, A):
        """
        Set the grid for the rectangle boundaries, and the data values.

        Parameters
        ----------
        x, y : 1D array-like, optional
            Monotonic arrays of length N+1 and M+1, respectively, specifying
            rectangle boundaries.  If not given, will default to
            ``range(N + 1)`` and ``range(M + 1)``, respectively.
        A : array-like
            The data to be color-coded. The interpretation depends on the
            shape:

            - (M, N) `~numpy.ndarray` or masked array: values to be colormapped
            - (M, N, 3): RGB array
            - (M, N, 4): RGBA array
        """
        A = self._normalize_image_array(A)
        x = np.arange(0.0, A.shape[1] + 1) if x is None else np.array(x, float).ravel()
        y = np.arange(0.0, A.shape[0] + 1) if y is None else np.array(y, float).ravel()
        if A.shape[:2] != (y.size - 1, x.size - 1):
            raise ValueError("Axes don't match array shape. Got %s, expected %s." % (A.shape[:2], (y.size - 1, x.size - 1)))
        if x[-1] < x[0]:
            x = x[::-1]
            A = A[:, ::-1]
        if y[-1] < y[0]:
            y = y[::-1]
            A = A[::-1]
        self._A = A
        self._Ax = x
        self._Ay = y
        self._imcache = None
        self.stale = True

    def set_array(self, *args):
        raise NotImplementedError('Method not supported')

    def get_cursor_data(self, event):
        x, y = (event.xdata, event.ydata)
        if x < self._Ax[0] or x > self._Ax[-1] or y < self._Ay[0] or (y > self._Ay[-1]):
            return None
        j = np.searchsorted(self._Ax, x) - 1
        i = np.searchsorted(self._Ay, y) - 1
        try:
            return self._A[i, j]
        except IndexError:
            return None