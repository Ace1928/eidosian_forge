from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots

    **deprecated**, use instead the plotly.graph_objects trace
    :class:`plotly.graph_objects.Violin`.

    :param (list|array) data: accepts either a list of numerical values,
        a list of dictionaries all with identical keys and at least one
        column of numeric values, or a pandas dataframe with at least one
        column of numbers.
    :param (str) data_header: the header of the data column to be used
        from an inputted pandas dataframe. Not applicable if 'data' is
        a list of numeric values.
    :param (str) group_header: applicable if grouping data by a variable.
        'group_header' must be set to the name of the grouping variable.
    :param (str|tuple|list|dict) colors: either a plotly scale name,
        an rgb or hex color, a color tuple, a list of colors or a
        dictionary. An rgb color is of the form 'rgb(x, y, z)' where
        x, y and z belong to the interval [0, 255] and a color tuple is a
        tuple of the form (a, b, c) where a, b and c belong to [0, 1].
        If colors is a list, it must contain valid color types as its
        members.
    :param (bool) use_colorscale: only applicable if grouping by another
        variable. Will implement a colorscale based on the first 2 colors
        of param colors. This means colors must be a list with at least 2
        colors in it (Plotly colorscales are accepted since they map to a
        list of two rgb colors). Default = False
    :param (dict) group_stats: a dictionary where each key is a unique
        value from the group_header column in data. Each value must be a
        number and will be used to color the violin plots if a colorscale
        is being used.
    :param (bool) rugplot: determines if a rugplot is draw on violin plot.
        Default = True
    :param (bool) sort: determines if violins are sorted
        alphabetically (True) or by input order (False). Default = False
    :param (float) height: the height of the violin plot.
    :param (float) width: the width of the violin plot.
    :param (str) title: the title of the violin plot.

    Example 1: Single Violin Plot

    >>> from plotly.figure_factory import create_violin
    >>> import plotly.graph_objs as graph_objects

    >>> import numpy as np
    >>> from scipy import stats

    >>> # create list of random values
    >>> data_list = np.random.randn(100)

    >>> # create violin fig
    >>> fig = create_violin(data_list, colors='#604d9e')

    >>> # plot
    >>> fig.show()

    Example 2: Multiple Violin Plots with Qualitative Coloring

    >>> from plotly.figure_factory import create_violin
    >>> import plotly.graph_objs as graph_objects

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy import stats

    >>> # create dataframe
    >>> np.random.seed(619517)
    >>> Nr=250
    >>> y = np.random.randn(Nr)
    >>> gr = np.random.choice(list("ABCDE"), Nr)
    >>> norm_params=[(0, 1.2), (0.7, 1), (-0.5, 1.4), (0.3, 1), (0.8, 0.9)]

    >>> for i, letter in enumerate("ABCDE"):
    ...     y[gr == letter] *=norm_params[i][1]+ norm_params[i][0]
    >>> df = pd.DataFrame(dict(Score=y, Group=gr))

    >>> # create violin fig
    >>> fig = create_violin(df, data_header='Score', group_header='Group',
    ...                    sort=True, height=600, width=1000)

    >>> # plot
    >>> fig.show()

    Example 3: Violin Plots with Colorscale

    >>> from plotly.figure_factory import create_violin
    >>> import plotly.graph_objs as graph_objects

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy import stats

    >>> # create dataframe
    >>> np.random.seed(619517)
    >>> Nr=250
    >>> y = np.random.randn(Nr)
    >>> gr = np.random.choice(list("ABCDE"), Nr)
    >>> norm_params=[(0, 1.2), (0.7, 1), (-0.5, 1.4), (0.3, 1), (0.8, 0.9)]

    >>> for i, letter in enumerate("ABCDE"):
    ...     y[gr == letter] *=norm_params[i][1]+ norm_params[i][0]
    >>> df = pd.DataFrame(dict(Score=y, Group=gr))

    >>> # define header params
    >>> data_header = 'Score'
    >>> group_header = 'Group'

    >>> # make groupby object with pandas
    >>> group_stats = {}
    >>> groupby_data = df.groupby([group_header])

    >>> for group in "ABCDE":
    ...     data_from_group = groupby_data.get_group(group)[data_header]
    ...     # take a stat of the grouped data
    ...     stat = np.median(data_from_group)
    ...     # add to dictionary
    ...     group_stats[group] = stat

    >>> # create violin fig
    >>> fig = create_violin(df, data_header='Score', group_header='Group',
    ...                     height=600, width=1000, use_colorscale=True,
    ...                     group_stats=group_stats)

    >>> # plot
    >>> fig.show()
    