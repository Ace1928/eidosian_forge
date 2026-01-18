import plotly.graph_objs as go
import pyviz_comms as comms
from param import concrete_descendents
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid
from .. import option_intersections
def _get_plot_state(self, element):
    fig_dict = plotly_renderer.get_plot_state(element)
    return fig_dict