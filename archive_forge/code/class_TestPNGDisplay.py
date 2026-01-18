from holoviews import Curve, Store
from holoviews.ipython import IPTestCase, notebook_extension
class TestPNGDisplay(TestDisplayHooks):

    def setUp(self):
        self.format = ['png']
        super().setUp()

    def test_store_render_png(self):
        curve = Curve([1, 2, 3])
        data, metadata = Store.render(curve)
        mime_types = {'image/png'}
        self.assertEqual(set(data), mime_types)