from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.subplots import make_subplots
import math
from numbers import Number
def _annotation_dict(text, lane, num_of_lanes, SUBPLOT_SPACING, row_col='col', flipped=True):
    l = (1 - (num_of_lanes - 1) * SUBPLOT_SPACING) / num_of_lanes
    if not flipped:
        xanchor = 'center'
        yanchor = 'middle'
        if row_col == 'col':
            x = (lane - 1) * (l + SUBPLOT_SPACING) + 0.5 * l
            y = 1.03
            textangle = 0
        elif row_col == 'row':
            y = (lane - 1) * (l + SUBPLOT_SPACING) + 0.5 * l
            x = 1.03
            textangle = 90
    elif row_col == 'col':
        xanchor = 'center'
        yanchor = 'bottom'
        x = (lane - 1) * (l + SUBPLOT_SPACING) + 0.5 * l
        y = 1.0
        textangle = 270
    elif row_col == 'row':
        xanchor = 'left'
        yanchor = 'middle'
        y = (lane - 1) * (l + SUBPLOT_SPACING) + 0.5 * l
        x = 1.0
        textangle = 0
    annotation_dict = dict(textangle=textangle, xanchor=xanchor, yanchor=yanchor, x=x, y=y, showarrow=False, xref='paper', yref='paper', text=str(text), font=dict(size=13, color=AXIS_TITLE_COLOR))
    return annotation_dict