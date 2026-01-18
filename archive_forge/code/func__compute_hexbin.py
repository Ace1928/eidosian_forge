from plotly.express._core import build_dataframe
from plotly.express._doc import make_docstring
from plotly.express._chart_types import choropleth_mapbox, scatter_mapbox
import numpy as np
import pandas as pd
def _compute_hexbin(x, y, x_range, y_range, color, nx, agg_func, min_count):
    """
    Computes the aggregation at hexagonal bin level.
    Also defines the coordinates of the hexagons for plotting.
    The binning is inspired by matplotlib's implementation.

    Parameters
    ----------
    x : np.ndarray
        Array of x values (shape N)
    y : np.ndarray
        Array of y values (shape N)
    x_range : np.ndarray
        Min and max x (shape 2)
    y_range : np.ndarray
        Min and max y (shape 2)
    color : np.ndarray
        Metric to aggregate at hexagon level (shape N)
    nx : int
        Number of hexagons horizontally
    agg_func : function
        Numpy compatible aggregator, this function must take a one-dimensional
        np.ndarray as input and output a scalar
    min_count : int
        Minimum number of points in the hexagon for the hexagon to be displayed

    Returns
    -------
    np.ndarray
        X coordinates of each hexagon (shape M x 6)
    np.ndarray
        Y coordinates of each hexagon (shape M x 6)
    np.ndarray
        Centers of the hexagons (shape M x 2)
    np.ndarray
        Aggregated value in each hexagon (shape M)

    """
    xmin = x_range.min()
    xmax = x_range.max()
    ymin = y_range.min()
    ymax = y_range.max()
    padding = 1e-09 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    Dx = xmax - xmin
    Dy = ymax - ymin
    if Dx == 0 and Dy > 0:
        dx = Dy / nx
    elif Dx == 0 and Dy == 0:
        dx, _ = _project_latlon_to_wgs84(1, 1)
    else:
        dx = Dx / nx
    dy = dx * np.sqrt(3)
    ny = np.ceil(Dy / dy).astype(int)
    ymin -= (ymin + dy * ny - ymax) / 2
    x = (x - xmin) / dx
    y = (y - ymin) / dy
    ix1 = np.round(x).astype(int)
    iy1 = np.round(y).astype(int)
    ix2 = np.floor(x).astype(int)
    iy2 = np.floor(y).astype(int)
    nx1 = nx + 1
    ny1 = ny + 1
    nx2 = nx
    ny2 = ny
    n = nx1 * ny1 + nx2 * ny2
    d1 = (x - ix1) ** 2 + 3.0 * (y - iy1) ** 2
    d2 = (x - ix2 - 0.5) ** 2 + 3.0 * (y - iy2 - 0.5) ** 2
    bdist = d1 < d2
    if color is None:
        lattice1 = np.zeros((nx1, ny1))
        lattice2 = np.zeros((nx2, ny2))
        c1 = (0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1) & bdist
        c2 = (0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2) & ~bdist
        np.add.at(lattice1, (ix1[c1], iy1[c1]), 1)
        np.add.at(lattice2, (ix2[c2], iy2[c2]), 1)
        if min_count is not None:
            lattice1[lattice1 < min_count] = np.nan
            lattice2[lattice2 < min_count] = np.nan
        accum = np.concatenate([lattice1.ravel(), lattice2.ravel()])
        good_idxs = ~np.isnan(accum)
    else:
        if min_count is None:
            min_count = 1
        lattice1 = np.empty((nx1, ny1), dtype=object)
        for i in range(nx1):
            for j in range(ny1):
                lattice1[i, j] = []
        lattice2 = np.empty((nx2, ny2), dtype=object)
        for i in range(nx2):
            for j in range(ny2):
                lattice2[i, j] = []
        for i in range(len(x)):
            if bdist[i]:
                if 0 <= ix1[i] < nx1 and 0 <= iy1[i] < ny1:
                    lattice1[ix1[i], iy1[i]].append(color[i])
            elif 0 <= ix2[i] < nx2 and 0 <= iy2[i] < ny2:
                lattice2[ix2[i], iy2[i]].append(color[i])
        for i in range(nx1):
            for j in range(ny1):
                vals = lattice1[i, j]
                if len(vals) >= min_count:
                    lattice1[i, j] = agg_func(vals)
                else:
                    lattice1[i, j] = np.nan
        for i in range(nx2):
            for j in range(ny2):
                vals = lattice2[i, j]
                if len(vals) >= min_count:
                    lattice2[i, j] = agg_func(vals)
                else:
                    lattice2[i, j] = np.nan
        accum = np.hstack((lattice1.astype(float).ravel(), lattice2.astype(float).ravel()))
        good_idxs = ~np.isnan(accum)
    agreggated_value = accum[good_idxs]
    centers = np.zeros((n, 2), float)
    centers[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
    centers[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
    centers[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
    centers[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
    centers[:, 0] *= dx
    centers[:, 1] *= dy
    centers[:, 0] += xmin
    centers[:, 1] += ymin
    centers = centers[good_idxs]
    hx = [0, 0.5, 0.5, 0, -0.5, -0.5]
    hy = [-0.5 / np.cos(np.pi / 6), -0.5 * np.tan(np.pi / 6), 0.5 * np.tan(np.pi / 6), 0.5 / np.cos(np.pi / 6), 0.5 * np.tan(np.pi / 6), -0.5 * np.tan(np.pi / 6)]
    m = len(centers)
    hxs = np.array([hx] * m) * dx + np.vstack(centers[:, 0])
    hys = np.array([hy] * m) * dy / np.sqrt(3) + np.vstack(centers[:, 1])
    return (hxs, hys, centers, agreggated_value)