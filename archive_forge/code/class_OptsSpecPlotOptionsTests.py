from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
class OptsSpecPlotOptionsTests(ComparisonTestCase):
    """
    Test the OptsSpec parser works correctly for plot options.
    """

    def test_plot_opts_simple(self):
        line = "Layout [fig_inches=(3,3) title='foo bar']"
        expected = {'Layout': {'plot': Options(title='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_with_space(self):
        """Space in the tuple, see issue #77"""
        line = "Layout [fig_inches=(3, 3) title='foo bar']"
        expected = {'Layout': {'plot': Options(title='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_simple_explicit(self):
        line = "Layout plot[fig_inches=(3,3) title='foo bar']"
        expected = {'Layout': {'plot': Options(title='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_with_space_explicit(self):
        line = "Layout plot[fig_inches=(3, 3) title='foo bar']"
        expected = {'Layout': {'plot': Options(title='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_dict_with_space(self):
        line = "Curve [fontsize={'xlabel': 10, 'title': 20}]"
        expected = {'Curve': {'plot': Options(fontsize={'xlabel': 10, 'title': 20})}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_dict_without_space(self):
        line = 'Curve [fontsize=dict(xlabel=10,title=20)]'
        expected = {'Curve': {'plot': Options(fontsize={'xlabel': 10, 'title': 20})}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_nested_brackets(self):
        line = "Curve [title=', '.join(('A', 'B'))]"
        expected = {'Curve': {'plot': Options(title='A, B')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_multiple_paths(self):
        line = "Image Curve [fig_inches=(3, 3) title='foo bar']"
        expected = {'Image': {'plot': Options(title='foo bar', fig_inches=(3, 3))}, 'Curve': {'plot': Options(title='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_plot_opts_multiple_paths_2(self):
        line = "Image Curve Layout Overlay[fig_inches=(3, 3) title='foo bar']"
        expected = {'Image': {'plot': Options(title='foo bar', fig_inches=(3, 3))}, 'Curve': {'plot': Options(title='foo bar', fig_inches=(3, 3))}, 'Layout': {'plot': Options(title='foo bar', fig_inches=(3, 3))}, 'Overlay': {'plot': Options(title='foo bar', fig_inches=(3, 3))}}
        self.assertEqual(OptsSpec.parse(line), expected)