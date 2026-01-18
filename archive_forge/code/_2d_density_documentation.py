from numbers import Number
import plotly.exceptions
import plotly.colors as clrs
from plotly.graph_objs import graph_objs

    **deprecated**, use instead
    :func:`plotly.express.density_heatmap`.

    :param (list|array) x: x-axis data for plot generation
    :param (list|array) y: y-axis data for plot generation
    :param (str|tuple|list) colorscale: either a plotly scale name, an rgb
        or hex color, a color tuple or a list or tuple of colors. An rgb
        color is of the form 'rgb(x, y, z)' where x, y, z belong to the
        interval [0, 255] and a color tuple is a tuple of the form
        (a, b, c) where a, b and c belong to [0, 1]. If colormap is a
        list, it must contain the valid color types aforementioned as its
        members.
    :param (int) ncontours: the number of 2D contours to draw on the plot
    :param (str) hist_color: the color of the plotted histograms
    :param (str) point_color: the color of the scatter points
    :param (str) point_size: the color of the scatter points
    :param (str) title: set the title for the plot
    :param (float) height: the height of the chart
    :param (float) width: the width of the chart

    Examples
    --------

    Example 1: Simple 2D Density Plot

    >>> from plotly.figure_factory import create_2d_density
    >>> import numpy as np

    >>> # Make data points
    >>> t = np.linspace(-1,1.2,2000)
    >>> x = (t**3)+(0.3*np.random.randn(2000))
    >>> y = (t**6)+(0.3*np.random.randn(2000))

    >>> # Create a figure
    >>> fig = create_2d_density(x, y)

    >>> # Plot the data
    >>> fig.show()

    Example 2: Using Parameters

    >>> from plotly.figure_factory import create_2d_density

    >>> import numpy as np

    >>> # Make data points
    >>> t = np.linspace(-1,1.2,2000)
    >>> x = (t**3)+(0.3*np.random.randn(2000))
    >>> y = (t**6)+(0.3*np.random.randn(2000))

    >>> # Create custom colorscale
    >>> colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)',
    ...              (1, 1, 0.2), (0.98,0.98,0.98)]

    >>> # Create a figure
    >>> fig = create_2d_density(x, y, colorscale=colorscale,
    ...       hist_color='rgb(255, 237, 222)', point_size=3)

    >>> # Plot the data
    >>> fig.show()
    