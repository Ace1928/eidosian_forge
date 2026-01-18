import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
class TestDimensioned_options(CustomBackendTestCase):

    def test_apply_options_current_backend_style(self):
        obj = ExampleElement([]).options(style_opt1='A')
        opts = Store.lookup_options('backend_1', obj, 'style')
        assert opts.options == {'style_opt1': 'A'}

    def test_apply_options_current_backend_style_invalid(self):
        err = "Unexpected option 'style_opt3' for ExampleElement type across all extensions. Similar options for current extension \\('backend_1'\\) are: \\['style_opt1', 'style_opt2'\\]\\."
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(style_opt3='A')

    def test_apply_options_current_backend_style_invalid_no_match(self):
        err = "Unexpected option 'zxy' for ExampleElement type across all extensions\\. No similar options found\\."
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(zxy='A')

    def test_apply_options_explicit_backend_style_invalid_cross_backend(self):
        err = "Unexpected option 'style_opt3' for ExampleElement type when using the 'backend_2' extension. Similar options are: \\['style_opt1', 'style_opt2'\\]\\."
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(style_opt3='A', backend='backend_2')

    def test_apply_options_explicit_backend_style_invalid_no_match(self):
        err = "Unexpected option 'zxy' for ExampleElement type when using the 'backend_2' extension. No similar options found\\."
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(zxy='A', backend='backend_2')

    def test_apply_options_current_backend_style_invalid_cross_backend_match(self):
        ExampleElement([]).options(plot_custom2='A')
        substr = "Option 'plot_custom2' for ExampleElement type not valid for selected backend ('backend_1'). Option only applies to following backends: ['backend_2']"
        self.log_handler.assertEndsWith('WARNING', substr)

    def test_apply_options_explicit_backend_style_invalid(self):
        err = "Unexpected option 'style_opt3' for ExampleElement type when using the 'backend_2' extension. Similar options are: \\['style_opt1', 'style_opt2'\\]\\."
        with self.assertRaisesRegex(ValueError, err):
            ExampleElement([]).options(style_opt3='A', backend='backend_2')

    def test_apply_options_current_backend_style_multiple(self):
        obj = ExampleElement([]).options(style_opt1='A', style_opt2='B')
        opts = Store.lookup_options('backend_1', obj, 'style')
        assert opts.options == {'style_opt1': 'A', 'style_opt2': 'B'}

    def test_apply_options_current_backend_plot(self):
        obj = ExampleElement([]).options(plot_opt1='A')
        opts = Store.lookup_options('backend_1', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A'}

    def test_apply_options_current_backend_plot_multiple(self):
        obj = ExampleElement([]).options(plot_opt1='A', plot_opt2='B')
        opts = Store.lookup_options('backend_1', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A', 'plot_opt2': 'B'}

    def test_apply_options_current_backend_plot_and_style(self):
        obj = ExampleElement([]).options(style_opt1='A', plot_opt1='B')
        plot_opts = Store.lookup_options('backend_1', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_1', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}

    def test_apply_options_explicit_backend_style(self):
        obj = ExampleElement([]).options(style_opt1='A', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'style')
        assert opts.options == {'style_opt1': 'A'}

    def test_apply_options_explicit_backend_style_multiple(self):
        obj = ExampleElement([]).options(style_opt1='A', style_opt2='B', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'style')
        assert opts.options == {'style_opt1': 'A', 'style_opt2': 'B'}

    def test_apply_options_explicit_backend_plot(self):
        obj = ExampleElement([]).options(plot_opt1='A', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A'}

    def test_apply_options_explicit_backend_plot_multiple(self):
        obj = ExampleElement([]).options(plot_opt1='A', plot_opt2='B', backend='backend_2')
        opts = Store.lookup_options('backend_2', obj, 'plot')
        assert opts.options == {'plot_opt1': 'A', 'plot_opt2': 'B'}

    def test_apply_options_explicit_backend_plot_and_style(self):
        obj = ExampleElement([]).options(style_opt1='A', plot_opt1='B', backend='backend_2')
        plot_opts = Store.lookup_options('backend_2', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_2', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}

    def test_apply_options_not_cloned(self):
        obj1 = ExampleElement([])
        obj2 = obj1.options(style_opt1='A', clone=False)
        opts = Store.lookup_options('backend_1', obj1, 'style')
        assert opts.options == {'style_opt1': 'A'}
        assert obj1 is obj2

    def test_apply_options_cloned(self):
        obj1 = ExampleElement([])
        obj2 = obj1.options(style_opt1='A')
        opts = Store.lookup_options('backend_1', obj2, 'style')
        assert opts.options == {'style_opt1': 'A'}
        assert obj1 is not obj2

    def test_apply_options_explicit_backend_persist_old_backend(self):
        obj = ExampleElement([])
        obj.opts(style_opt1='A', plot_opt1='B', backend='backend_1')
        obj.opts(style_opt1='C', plot_opt1='D', backend='backend_2')
        plot_opts = Store.lookup_options('backend_1', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_1', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}
        plot_opts = Store.lookup_options('backend_2', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'D'}
        style_opts = Store.lookup_options('backend_2', obj, 'style')
        assert style_opts.options == {'style_opt1': 'C'}

    def test_apply_options_explicit_backend_persists_other_backend_inverted(self):
        obj = ExampleElement([])
        obj.opts(style_opt1='A', plot_opt1='B', backend='backend_2')
        obj.opts(style_opt1='C', plot_opt1='D', backend='backend_1')
        plot_opts = Store.lookup_options('backend_1', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'D'}
        style_opts = Store.lookup_options('backend_1', obj, 'style')
        assert style_opts.options == {'style_opt1': 'C'}
        plot_opts = Store.lookup_options('backend_2', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'B'}
        style_opts = Store.lookup_options('backend_2', obj, 'style')
        assert style_opts.options == {'style_opt1': 'A'}

    def test_apply_options_when_backend_switched(self):
        obj = ExampleElement([])
        Store.current_backend = 'backend_2'
        obj.opts(style_opt1='A', plot_opt1='B')
        Store.current_backend = 'backend_1'
        obj.opts(style_opt1='C', plot_opt1='D', backend='backend_2')
        plot_opts = Store.lookup_options('backend_2', obj, 'plot')
        assert plot_opts.options == {'plot_opt1': 'D'}
        style_opts = Store.lookup_options('backend_2', obj, 'style')
        assert style_opts.options == {'style_opt1': 'C'}