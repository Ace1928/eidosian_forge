from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.subplots import make_subplots
import math
from numbers import Number
def _legend_annotation(color_name):
    legend_title = dict(textangle=0, xanchor='left', yanchor='middle', x=LEGEND_ANNOT_X, y=1.03, showarrow=False, xref='paper', yref='paper', text='factor({})'.format(color_name), font=dict(size=13, color='#000000'))
    return legend_title