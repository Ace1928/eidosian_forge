import numpy as np
from holoviews import Curve, HoloMap, Image, Overlay
from holoviews.core.options import Store, StoreOptions
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl  # noqa Register backend
class TestStoreOptionsMerge(ComparisonTestCase):

    def setUp(self):
        Store.current_backend = 'matplotlib'
        self.expected = {'Image': {'plot': {'fig_size': 150}, 'style': {'cmap': 'Blues'}}}

    def test_full_spec_format(self):
        out = StoreOptions.merge_options(['plot', 'style'], options={'Image': {'plot': dict(fig_size=150), 'style': dict(cmap='Blues')}})
        self.assertEqual(out, self.expected)

    def test_options_partitioned_format(self):
        out = StoreOptions.merge_options(['plot', 'style'], options=dict(plot={'Image': dict(fig_size=150)}, style={'Image': dict(cmap='Blues')}))
        self.assertEqual(out, self.expected)

    def test_partitioned_format(self):
        out = StoreOptions.merge_options(['plot', 'style'], plot={'Image': dict(fig_size=150)}, style={'Image': dict(cmap='Blues')})
        self.assertEqual(out, self.expected)