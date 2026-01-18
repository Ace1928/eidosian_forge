from traitlets.config.configurable import SingletonConfigurable
from traitlets import (
class InlineBackend(InlineBackendConfig):
    """An object to store configuration of the inline backend."""
    rc = Dict({}, help="Dict to manage matplotlib configuration defaults in the inline\n        backend. As of v0.1.4 IPython/Jupyter do not override defaults out of\n        the box, but third-party tools may use it to manage rc data. To change\n        personal defaults for matplotlib,  use matplotlib's configuration\n        tools, or customize this class in your `ipython_config.py` file for\n        IPython/Jupyter-specific usage.").tag(config=True)
    figure_formats = Set({'png'}, help="A set of figure formats to enable: 'png',\n                'retina', 'jpeg', 'svg', 'pdf'.").tag(config=True)

    def _update_figure_formatters(self):
        if self.shell is not None:
            from IPython.core.pylabtools import select_figure_formats
            select_figure_formats(self.shell, self.figure_formats, **self.print_figure_kwargs)

    def _figure_formats_changed(self, name, old, new):
        if 'jpg' in new or 'jpeg' in new:
            if not pil_available():
                raise TraitError('Requires PIL/Pillow for JPG figures')
        self._update_figure_formatters()
    figure_format = Unicode(help='The figure format to enable (deprecated\n                                         use `figure_formats` instead)').tag(config=True)

    def _figure_format_changed(self, name, old, new):
        if new:
            self.figure_formats = {new}
    print_figure_kwargs = Dict({'bbox_inches': 'tight'}, help='Extra kwargs to be passed to fig.canvas.print_figure.\n\n        Logical examples include: bbox_inches, quality (for jpeg figures), etc.\n        ').tag(config=True)
    _print_figure_kwargs_changed = _update_figure_formatters
    close_figures = Bool(True, help='Close all figures at the end of each cell.\n\n        When True, ensures that each cell starts with no active figures, but it\n        also means that one must keep track of references in order to edit or\n        redraw figures in subsequent cells. This mode is ideal for the notebook,\n        where residual plots from other cells might be surprising.\n\n        When False, one must call figure() to create new figures. This means\n        that gcf() and getfigs() can reference figures created in other cells,\n        and the active figure can continue to be edited with pylab/pyplot\n        methods that reference the current active figure. This mode facilitates\n        iterative editing of figures, and behaves most consistently with\n        other matplotlib backends, but figure barriers between cells must\n        be explicit.\n        ').tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)