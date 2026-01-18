from holoviews import Curve, Store
from holoviews.ipython import IPTestCase, notebook_extension
class TestSVGDisplay(TestDisplayHooks):

    def setUp(self):
        self.format = ['svg']
        super().setUp()

    def test_store_render_svg(self):
        curve = Curve([1, 2, 3])
        data, metadata = Store.render(curve)
        mime_types = {'image/svg+xml'}
        self.assertEqual(set(data), mime_types)