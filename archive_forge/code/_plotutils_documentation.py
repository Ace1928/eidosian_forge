import numpy as np
from scipy._lib.decorator import decorator as _decorator

    Plot the given Voronoi diagram in 2-D

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points : bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha : float, optional
        Specifies the line alpha for polygon boundaries
    point_size : float, optional
        Specifies the size of points

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Voronoi

    Notes
    -----
    Requires Matplotlib.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import Voronoi, voronoi_plot_2d

    Create a set of points for the example:

    >>> rng = np.random.default_rng()
    >>> points = rng.random((10,2))

    Generate the Voronoi diagram for the points:

    >>> vor = Voronoi(points)

    Use `voronoi_plot_2d` to plot the diagram:

    >>> fig = voronoi_plot_2d(vor)

    Use `voronoi_plot_2d` to plot the diagram again, with some settings
    customized:

    >>> fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
    ...                       line_width=2, line_alpha=0.6, point_size=2)
    >>> plt.show()

    