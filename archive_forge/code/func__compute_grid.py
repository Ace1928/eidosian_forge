import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _compute_grid(coordinates, values, interp_mode='ilr'):
    """
    Transform data points with Cartesian or ILR mapping, then Compute
    interpolation on a regular grid.

    Parameters
    ==========

    coordinates : array-like
        Barycentric coordinates of data points.
    values : 1-d array-like
        Data points, field to be represented as contours.
    interp_mode : 'ilr' (default) or 'cartesian'
        Defines how data are interpolated to compute contours.
    """
    if interp_mode == 'cartesian':
        M, invM = _transform_barycentric_cartesian()
        coord_points = np.einsum('ik, kj -> ij', M, coordinates)
    elif interp_mode == 'ilr':
        coordinates = _replace_zero_coords(coordinates)
        coord_points = _ilr_transform(coordinates)
    else:
        raise ValueError('interp_mode should be cartesian or ilr')
    xx, yy = coord_points[:2]
    x_min, x_max = (xx.min(), xx.max())
    y_min, y_max = (yy.min(), yy.max())
    n_interp = max(200, int(np.sqrt(len(values))))
    gr_x = np.linspace(x_min, x_max, n_interp)
    gr_y = np.linspace(y_min, y_max, n_interp)
    grid_x, grid_y = np.meshgrid(gr_x, gr_y)
    grid_z = scipy_interp.griddata(coord_points[:2].T, values, (grid_x, grid_y), method='cubic')
    grid_z_other = scipy_interp.griddata(coord_points[:2].T, values, (grid_x, grid_y), method='nearest')
    return (grid_z, gr_x, gr_y)