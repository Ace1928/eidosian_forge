from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
class OptsSpecStyleOptionsTests(ComparisonTestCase):

    def test_style_opts_simple(self):
        line = "Layout (string='foo')"
        expected = {'Layout': {'style': Options(string='foo')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_simple_explicit(self):
        line = "Layout style(string='foo')"
        expected = {'Layout': {'style': Options(string='foo')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_intermediate(self):
        line = "Layout (string='foo' test=3, b=True)"
        expected = {'Layout': {'style': Options(string='foo', test=3, b=True)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_intermediate_explicit(self):
        line = "Layout style(string='foo' test=3, b=True )"
        expected = {'Layout': {'style': Options(string='foo', test=3, b=True)}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_advanced(self):
        line = "Layout (string='foo' test=3, b=True color=Cycle(values=[1,2]))"
        expected = {'Layout': {'style': Options(string='foo', test=3, b=True, color=Cycle(values=[1, 2]))}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_dict_with_space(self):
        line = "Curve (fontsize={'xlabel': 10, 'title': 20})"
        expected = {'Curve': {'style': Options(fontsize={'xlabel': 10, 'title': 20})}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_dict_without_space(self):
        line = "Curve (fontsize={'xlabel': 10,'title': 20})"
        expected = {'Curve': {'style': Options(fontsize={'xlabel': 10, 'title': 20})}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_style_opts_cycle_function(self):
        import numpy as np
        np.random.seed(42)
        line = 'Curve (color=Cycle(values=list(np.random.rand(3,3))))'
        options = OptsSpec.parse(line, {'np': np, 'Cycle': Cycle})
        self.assertTrue('Curve' in options)
        self.assertTrue('style' in options['Curve'])
        self.assertTrue('color' in options['Curve']['style'].kwargs)
        self.assertTrue(isinstance(options['Curve']['style'].kwargs['color'], Cycle))
        values = np.array([[0.37454012, 0.95071431, 0.73199394], [0.59865848, 0.15601864, 0.15599452], [0.05808361, 0.86617615, 0.60111501]])
        self.assertEqual(np.array(options['Curve']['style'].kwargs['color'].values), values)

    def test_style_opts_cycle_list(self):
        line = "Curve (color=Cycle(values=['r', 'g', 'b']))"
        expected = {'Curve': {'style': Options(color=Cycle(values=['r', 'g', 'b']))}}
        self.assertEqual(OptsSpec.parse(line, {'Cycle': Cycle}), expected)

    def test_style_opts_multiple_paths(self):
        line = "Image Curve (color='beautiful')"
        expected = {'Image': {'style': Options(color='beautiful')}, 'Curve': {'style': Options(color='beautiful')}}
        self.assertEqual(OptsSpec.parse(line), expected)