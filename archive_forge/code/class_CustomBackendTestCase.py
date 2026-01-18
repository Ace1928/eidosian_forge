import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
class CustomBackendTestCase(LoggingComparisonTestCase):
    """
    Registers fake backends with the Store to test options on.
    """

    def setUp(self):
        super().setUp()
        self.current_backend = Store.current_backend
        self.register_custom(ExampleElement, 'backend_1', ['plot_custom1'])
        self.register_custom(ExampleElement, 'backend_2', ['plot_custom2'])
        Store.set_current_backend('backend_1')

    def tearDown(self):
        super().tearDown()
        Store._weakrefs = {}
        Store._options.pop('backend_1')
        Store._options.pop('backend_2')
        Store._custom_options.pop('backend_1')
        Store._custom_options.pop('backend_2')
        Store.set_current_backend(self.current_backend)
        Store.renderers.pop('backend_1')
        Store.renderers.pop('backend_2')

    @classmethod
    def register_custom(cls, objtype, backend, custom_plot=None, custom_style=None):
        if custom_style is None:
            custom_style = []
        if custom_plot is None:
            custom_plot = []
        groups = Options._option_groups
        if backend not in Store._options:
            Store._options[backend] = OptionTree([], groups=groups)
            Store._custom_options[backend] = {}
        name = objtype.__name__
        style_opts = Keywords(['style_opt1', 'style_opt2'] + custom_style, name)
        plot_opts = Keywords(['plot_opt1', 'plot_opt2'] + custom_plot, name)
        opt_groups = {'plot': Options(allowed_keywords=plot_opts), 'style': Options(allowed_keywords=style_opts), 'output': Options(allowed_keywords=['backend'])}
        Store._options[backend][name] = opt_groups
        Store.renderers[backend] = MockRenderer(backend)