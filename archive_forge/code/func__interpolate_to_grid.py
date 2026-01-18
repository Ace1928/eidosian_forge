import numpy as np
def _interpolate_to_grid(nx, ny, x, y, *scalars, **kwargs):
    """
    Interpolate two vector components and zero or more scalar fields,
    which can be irregular, to a regular grid.

    Parameters
    ----------
    nx
        Number of points at which to interpolate in x direction.
    ny
        Number of points at which to interpolate in y direction.
    x
        Array of source points in x direction.
    y
        Array of source points in y direction.

    Other Parameters
    ----------------
    scalars
        Zero or more scalar fields to regrid along with the vector
        components.
    target_extent
        The extent in the target CRS that the grid should occupy, in the
        form ``(x-lower, x-upper, y-lower, y-upper)``. Defaults to cover
        the full extent of the vector field.

    """
    target_extent = kwargs.get('target_extent', None)
    if target_extent is None:
        target_extent = (x.min(), x.max(), y.min(), y.max())
    x0, x1, y0, y1 = target_extent
    xr = x1 - x0
    yr = y1 - y0
    points = np.column_stack([(x.ravel() - x0) / xr, (y.ravel() - y0) / yr])
    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    s_grid_tuple = tuple()
    for s in scalars:
        s_grid_tuple += (griddata(points, s.ravel(), (x_grid, y_grid), method='linear'),)
    return (x_grid * xr + x0, y_grid * yr + y0) + s_grid_tuple