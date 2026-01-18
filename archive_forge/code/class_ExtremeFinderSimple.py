import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
class ExtremeFinderSimple:
    """
    A helper class to figure out the range of grid lines that need to be drawn.
    """

    def __init__(self, nx, ny):
        """
        Parameters
        ----------
        nx, ny : int
            The number of samples in each direction.
        """
        self.nx = nx
        self.ny = ny

    def __call__(self, transform_xy, x1, y1, x2, y2):
        """
        Compute an approximation of the bounding box obtained by applying
        *transform_xy* to the box delimited by ``(x1, y1, x2, y2)``.

        The intended use is to have ``(x1, y1, x2, y2)`` in axes coordinates,
        and have *transform_xy* be the transform from axes coordinates to data
        coordinates; this method then returns the range of data coordinates
        that span the actual axes.

        The computation is done by sampling ``nx * ny`` equispaced points in
        the ``(x1, y1, x2, y2)`` box and finding the resulting points with
        extremal coordinates; then adding some padding to take into account the
        finite sampling.

        As each sampling step covers a relative range of *1/nx* or *1/ny*,
        the padding is computed by expanding the span covered by the extremal
        coordinates by these fractions.
        """
        x, y = np.meshgrid(np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        xt, yt = transform_xy(np.ravel(x), np.ravel(y))
        return self._add_pad(xt.min(), xt.max(), yt.min(), yt.max())

    def _add_pad(self, x_min, x_max, y_min, y_max):
        """Perform the padding mentioned in `__call__`."""
        dx = (x_max - x_min) / self.nx
        dy = (y_max - y_min) / self.ny
        return (x_min - dx, x_max + dx, y_min - dy, y_max + dy)