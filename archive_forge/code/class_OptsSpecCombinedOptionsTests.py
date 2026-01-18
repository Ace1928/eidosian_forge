from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
class OptsSpecCombinedOptionsTests(ComparisonTestCase):

    def test_combined_1(self):
        line = "Layout plot[fig_inches=(3,3) foo='bar baz'] Layout (string='foo')"
        expected = {'Layout': {'plot': Options(foo='bar baz', fig_inches=(3, 3)), 'style': Options(string='foo')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_combined_two_types_1(self):
        line = "Layout plot[fig_inches=(3,3) foo='bar baz'] Image (string='foo')"
        expected = {'Layout': {'plot': Options(foo='bar baz', fig_inches=(3, 3))}, 'Image': {'style': Options(string='foo')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_combined_two_types_2(self):
        line = "Layout plot[fig_inches=(3, 3)] Image (string='foo') [foo='bar baz']"
        expected = {'Layout': {'plot': Options(fig_inches=(3, 3))}, 'Image': {'style': Options(string='foo'), 'plot': Options(foo='bar baz')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_combined_multiple_paths(self):
        line = "Image Curve {+framewise} [fig_inches=(3, 3) title='foo bar'] (c='b') Layout [string='foo'] Overlay"
        expected = {'Image': {'norm': Options(framewise=True, axiswise=False), 'plot': Options(title='foo bar', fig_inches=(3, 3)), 'style': Options(c='b')}, 'Curve': {'norm': Options(framewise=True, axiswise=False), 'plot': Options(title='foo bar', fig_inches=(3, 3)), 'style': Options(c='b')}, 'Layout': {'plot': Options(string='foo')}, 'Overlay': {}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_combined_multiple_paths_merge(self):
        line = "Image Curve [fig_inches=(3, 3)] (c='b') Image (s=3)"
        expected = {'Image': {'plot': Options(fig_inches=(3, 3)), 'style': Options(c='b', s=3)}, 'Curve': {'plot': Options(fig_inches=(3, 3)), 'style': Options(c='b')}}
        self.assertEqual(OptsSpec.parse(line), expected)

    def test_combined_multiple_paths_merge_precedence(self):
        line = "Image (s=0, c='b') Image (s=3)"
        expected = {'Image': {'style': Options(c='b', s=3)}}
        self.assertEqual(OptsSpec.parse(line), expected)