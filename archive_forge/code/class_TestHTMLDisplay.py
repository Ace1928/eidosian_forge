from holoviews import Curve, Store
from holoviews.ipython import IPTestCase, notebook_extension
class TestHTMLDisplay(TestDisplayHooks):

    def setUp(self):
        self.format = ['html']
        super().setUp()

    def test_store_render_html(self):
        curve = Curve([1, 2, 3])
        data, metadata = Store.render(curve)
        mime_types = {'text/html'}
        self.assertEqual(set(data), mime_types)