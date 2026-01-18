from unittest import SkipTest
import holoviews as hv
from holoviews.core.options import Store
from pyviz_comms import CommManager
from holoviews.operation import Compositor
class TestOptsMagic(ExtensionTestCase):

    def setUp(self):
        super().setUp()
        self.cell('import numpy as np')
        self.cell('from holoviews import DynamicMap, Curve, Image')

    def tearDown(self):
        Store.custom_options(val={})
        super().tearDown()

    def test_cell_opts_style(self):
        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")
        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', " Image (cmap='hot')", 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)
        assert 0 in Store.custom_options(), 'Custom OptionTree creation failed'
        self.assertEqual(Store.lookup_options('matplotlib', self.get_object('mat1'), 'style').options.get('cmap', None), 'hot')

    def test_cell_opts_style_dynamic(self):
        self.cell("dmap = DynamicMap(lambda X: Curve(np.random.rand(5,2), name='dmap'), kdims=['x']).redim.range(x=(0, 10)).opts({'Curve': dict(linewidth=2, color='black')})")
        self.assertEqual(self.get_object('dmap').id, None)
        self.cell_magic('opts', ' Curve (linewidth=3 alpha=0.5)', 'dmap')
        self.assertEqual(self.get_object('dmap').id, 0)
        assert 0 in Store.custom_options(), 'Custom OptionTree creation failed'
        opts = Store.lookup_options('matplotlib', self.get_object('dmap')[0], 'style').options
        self.assertEqual(opts, {'linewidth': 3, 'alpha': 0.5, 'color': 'black'})

    def test_cell_opts_plot_float_division(self):
        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")
        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', ' Image [aspect=3/4]', 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)
        assert 0 in Store.custom_options(), 'Custom OptionTree creation failed'
        self.assertEqual(Store.lookup_options('matplotlib', self.get_object('mat1'), 'plot').options.get('aspect', False), 3 / 4.0)

    def test_cell_opts_plot(self):
        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")
        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', ' Image [show_title=False]', 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)
        assert 0 in Store.custom_options(), 'Custom OptionTree creation failed'
        self.assertEqual(Store.lookup_options('matplotlib', self.get_object('mat1'), 'plot').options.get('show_title', True), False)

    def test_cell_opts_plot_dynamic(self):
        self.cell("dmap = DynamicMap(lambda X: Image(np.random.rand(5,5), name='dmap'), kdims=['x']).redim.range(x=(0, 10)).opts({'Image': dict(xaxis='top', xticks=3)})")
        self.assertEqual(self.get_object('dmap').id, None)
        self.cell_magic('opts', " Image [xaxis=None yaxis='right']", 'dmap')
        self.assertEqual(self.get_object('dmap').id, 0)
        assert 0 in Store.custom_options(), 'Custom OptionTree creation failed'
        opts = Store.lookup_options('matplotlib', self.get_object('dmap')[0], 'plot').options
        self.assertEqual(opts, {'xticks': 3, 'xaxis': None, 'yaxis': 'right'})

    def test_cell_opts_norm(self):
        self.cell("mat1 = Image(np.random.rand(5,5), name='mat1')")
        self.assertEqual(self.get_object('mat1').id, None)
        self.cell_magic('opts', ' Image {+axiswise}', 'mat1')
        self.assertEqual(self.get_object('mat1').id, 0)
        assert 0 in Store.custom_options(), 'Custom OptionTree creation failed'
        self.assertEqual(Store.lookup_options('matplotlib', self.get_object('mat1'), 'norm').options.get('axiswise', True), True)