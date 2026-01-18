import plotly.graph_objs as go
import pyviz_comms as comms
from param import concrete_descendents
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.plotly.element import ElementPlot
from holoviews.plotting.plotly.util import figure_grid
from .. import option_intersections
class TestPlotlyPlot(ComparisonTestCase):

    def setUp(self):
        self.previous_backend = Store.current_backend
        Store.set_current_backend('plotly')
        self.comm_manager = plotly_renderer.comm_manager
        plotly_renderer.comm_manager = comms.CommManager
        self._padding = {}
        for plot in concrete_descendents(ElementPlot).values():
            self._padding[plot] = plot.padding
            plot.padding = 0

    def tearDown(self):
        Store.current_backend = self.previous_backend
        plotly_renderer.comm_manager = self.comm_manager
        for plot, padding in self._padding.items():
            plot.padding = padding

    def _get_plot_state(self, element):
        fig_dict = plotly_renderer.get_plot_state(element)
        return fig_dict

    def assert_property_values(self, obj, props):
        """
        Assert that a dictionary has the specified properties, handling magic underscore
        notation

        For example

        self.assert_property_values(
            {'a': {'b': 23}, 'c': 42},
            {'a_b': 23, 'c': 42}
        )

        will pass this test
        """
        for prop, val in props.items():
            prop_parts = prop.split('_')
            prop_parent = obj
            for prop_part in prop_parts[:-1]:
                prop_parent = prop_parent.get(prop_part, {})
            self.assertEqual(val, prop_parent[prop_parts[-1]])