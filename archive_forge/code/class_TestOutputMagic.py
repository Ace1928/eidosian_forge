from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
class TestOutputMagic(ExtensionTestCase):

    def tearDown(self):
        super().tearDown()

    def test_output_svg(self):
        self.line_magic('output', "fig='svg'")
        self.assertEqual(hv.util.OutputSettings.options.get('fig', None), 'svg')

    def test_output_holomap_scrubber(self):
        self.line_magic('output', "holomap='scrubber'")
        self.assertEqual(hv.util.OutputSettings.options.get('holomap', None), 'scrubber')

    def test_output_holomap_widgets(self):
        self.line_magic('output', "holomap='widgets'")
        self.assertEqual(hv.util.OutputSettings.options.get('holomap', None), 'widgets')

    def test_output_widgets_live(self):
        self.line_magic('output', "widgets='live'")
        self.assertEqual(hv.util.OutputSettings.options.get('widgets', None), 'live')

    def test_output_fps(self):
        self.line_magic('output', 'fps=100')
        self.assertEqual(hv.util.OutputSettings.options.get('fps', None), 100)

    def test_output_size(self):
        self.line_magic('output', 'size=50')
        self.assertEqual(hv.util.OutputSettings.options.get('size', None), 50)

    def test_output_invalid_size(self):
        self.line_magic('output', 'size=-50')
        self.assertEqual(hv.util.OutputSettings.options.get('size', None), None)