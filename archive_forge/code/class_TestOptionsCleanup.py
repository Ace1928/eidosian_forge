import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
class TestOptionsCleanup(CustomBackendTestCase):

    def test_opts_resassignment_cleans_unused_tree(self):
        obj = ExampleElement([]).opts(style_opt1='A').opts(plot_opt1='B')
        custom_options = Store._custom_options['backend_1']
        self.assertIn(obj.id, custom_options)
        self.assertEqual(len(custom_options), 1)

    def test_opts_multiple_resassignment_cleans_unused_tree(self):
        obj = HoloMap({0: ExampleElement([]), 1: ExampleElement([])}).opts(style_opt1='A').opts(plot_opt1='B')
        custom_options = Store._custom_options['backend_1']
        self.assertIn(obj.last.id, custom_options)
        self.assertEqual(len(custom_options), 2)
        for o in obj:
            o.id = None
        self.assertEqual(len(custom_options), 0)

    def test_opts_resassignment_cleans_unused_tree_cross_backend(self):
        obj = ExampleElement([]).opts(style_opt1='A').opts(plot_opt1='B', backend='backend_2')
        custom_options = Store._custom_options['backend_1']
        self.assertIn(obj.id, custom_options)
        self.assertEqual(len(custom_options), 1)
        custom_options = Store._custom_options['backend_2']
        self.assertIn(obj.id, custom_options)
        self.assertEqual(len(custom_options), 1)

    def test_garbage_collect_cleans_unused_tree(self):
        obj = ExampleElement([]).opts(style_opt1='A')
        del obj
        gc.collect()
        custom_options = Store._custom_options['backend_1']
        self.assertEqual(len(custom_options), 0)

    def test_partial_garbage_collect_does_not_clear_tree(self):
        obj = HoloMap({0: ExampleElement([]), 1: ExampleElement([])}).opts(style_opt1='A')
        obj.pop(0)
        gc.collect()
        custom_options = Store._custom_options['backend_1']
        self.assertIn(obj.last.id, custom_options)
        self.assertEqual(len(custom_options), 1)
        obj.pop(1)
        gc.collect()
        self.assertEqual(len(custom_options), 0)

    def test_opts_clear_cleans_unused_tree(self):
        ExampleElement([]).opts(style_opt1='A').opts.clear()
        custom_options = Store._custom_options['backend_1']
        self.assertEqual(len(custom_options), 0)