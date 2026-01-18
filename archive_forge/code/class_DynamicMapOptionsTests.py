import time
import uuid
from collections import deque
import numpy as np
import param
import pytest
from holoviews import Dimension, GridSpace, Layout, NdLayout, NdOverlay
from holoviews.core.options import Store
from holoviews.core.spaces import Callable, DynamicMap, HoloMap
from holoviews.element import Curve, Image, Points, Scatter, Text
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import histogram
from holoviews.plotting.util import initialize_dynamic
from holoviews.streams import (
from holoviews.util import Dynamic
from ..utils import LoggingComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
class DynamicMapOptionsTests(CustomBackendTestCase):

    def test_dynamic_options(self):
        dmap = DynamicMap(lambda X: ExampleElement(None), kdims=['X']).redim.range(X=(0, 10))
        dmap = dmap.options(plot_opt1='red')
        opts = Store.lookup_options('backend_1', dmap[0], 'plot')
        self.assertEqual(opts.options, {'plot_opt1': 'red'})

    def test_dynamic_options_no_clone(self):
        dmap = DynamicMap(lambda X: ExampleElement(None), kdims=['X']).redim.range(X=(0, 10))
        dmap.options(plot_opt1='red', clone=False)
        opts = Store.lookup_options('backend_1', dmap[0], 'plot')
        self.assertEqual(opts.options, {'plot_opt1': 'red'})

    def test_dynamic_opts_link_inputs(self):
        stream = LinkedStream()
        inputs = [DynamicMap(lambda: None, streams=[stream])]
        dmap = DynamicMap(Callable(lambda X: ExampleElement(None), inputs=inputs), kdims=['X']).redim.range(X=(0, 10))
        styled_dmap = dmap.options(plot_opt1='red', clone=False)
        opts = Store.lookup_options('backend_1', dmap[0], 'plot')
        self.assertEqual(opts.options, {'plot_opt1': 'red'})
        self.assertIs(styled_dmap, dmap)
        self.assertTrue(dmap.callback.link_inputs)
        unstyled_dmap = dmap.callback.inputs[0].callback.inputs[0]
        opts = Store.lookup_options('backend_1', unstyled_dmap[0], 'plot')
        self.assertEqual(opts.options, {})
        original_dmap = unstyled_dmap.callback.inputs[0]
        self.assertIs(stream, original_dmap.streams[0])