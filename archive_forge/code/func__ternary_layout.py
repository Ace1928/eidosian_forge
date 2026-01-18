import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _ternary_layout(title='Ternary contour plot', width=550, height=525, pole_labels=['a', 'b', 'c']):
    """
    Layout of ternary contour plot, to be passed to ``go.FigureWidget``
    object.

    Parameters
    ==========
    title : str or None
        Title of ternary plot
    width : int
        Figure width.
    height : int
        Figure height.
    pole_labels : str, default ['a', 'b', 'c']
        Names of the three poles of the triangle.
    """
    return dict(title=title, width=width, height=height, ternary=dict(sum=1, aaxis=dict(title=dict(text=pole_labels[0]), min=0.01, linewidth=2, ticks='outside'), baxis=dict(title=dict(text=pole_labels[1]), min=0.01, linewidth=2, ticks='outside'), caxis=dict(title=dict(text=pole_labels[2]), min=0.01, linewidth=2, ticks='outside')), showlegend=False)