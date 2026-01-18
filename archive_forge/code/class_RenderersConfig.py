import textwrap
from copy import copy
import os
from packaging.version import Version
from plotly import optional_imports
from plotly.io._base_renderers import (
from plotly.io._utils import validate_coerce_fig_to_dict
class RenderersConfig(object):
    """
    Singleton object containing the current renderer configurations
    """

    def __init__(self):
        self._renderers = {}
        self._default_name = None
        self._default_renderers = []
        self._render_on_display = False
        self._to_activate = []

    def __len__(self):
        return len(self._renderers)

    def __contains__(self, item):
        return item in self._renderers

    def __iter__(self):
        return iter(self._renderers)

    def __getitem__(self, item):
        renderer = self._renderers[item]
        return renderer

    def __setitem__(self, key, value):
        if not isinstance(value, (MimetypeRenderer, ExternalRenderer)):
            raise ValueError('Renderer must be a subclass of MimetypeRenderer or ExternalRenderer.\n    Received value with type: {typ}'.format(typ=type(value)))
        self._renderers[key] = value

    def __delitem__(self, key):
        del self._renderers[key]
        if self._default == key:
            self._default = None

    def keys(self):
        return self._renderers.keys()

    def items(self):
        return self._renderers.items()

    def update(self, d={}, **kwargs):
        """
        Update one or more renderers from a dict or from input keyword
        arguments.

        Parameters
        ----------
        d: dict
            Dictionary from renderer names to new renderer objects.

        kwargs
            Named argument value pairs where the name is a renderer name
            and the value is a new renderer object
        """
        for k, v in dict(d, **kwargs).items():
            self[k] = v

    @property
    def default(self):
        """
        The default renderer, or None if no there is no default

        If not None, the default renderer is used to render
        figures when the `plotly.io.show` function is called on a Figure.

        If `plotly.io.renderers.render_on_display` is True, then the default
        renderer will also be used to display Figures automatically when
        displayed in the Jupyter Notebook

        Multiple renderers may be registered by separating their names with
        '+' characters. For example, to specify rendering compatible with
        the classic Jupyter Notebook, JupyterLab, and PDF export:

        >>> import plotly.io as pio
        >>> pio.renderers.default = 'notebook+jupyterlab+pdf'

        The names of available renderers may be retrieved with:

        >>> import plotly.io as pio
        >>> list(pio.renderers)

        Returns
        -------
        str
        """
        return self._default_name

    @default.setter
    def default(self, value):
        if not value:
            self._default_name = ''
            self._default_renderers = []
            return
        renderer_names = self._validate_coerce_renderers(value)
        self._default_name = value
        self._default_renderers = [self[name] for name in renderer_names]
        self._to_activate = list(self._default_renderers)

    @property
    def render_on_display(self):
        """
        If True, the default mimetype renderers will be used to render
        figures when they are displayed in an IPython context.

        Returns
        -------
        bool
        """
        return self._render_on_display

    @render_on_display.setter
    def render_on_display(self, val):
        self._render_on_display = bool(val)

    def _activate_pending_renderers(self, cls=object):
        """
        Activate all renderers that are waiting in the _to_activate list

        Parameters
        ----------
        cls
            Only activate renders that are subclasses of this class
        """
        to_activate_with_cls = [r for r in self._to_activate if cls and isinstance(r, cls)]
        while to_activate_with_cls:
            renderer = to_activate_with_cls.pop(0)
            renderer.activate()
        self._to_activate = [r for r in self._to_activate if not (cls and isinstance(r, cls))]

    def _validate_coerce_renderers(self, renderers_string):
        """
        Input a string and validate that it contains the names of one or more
        valid renderers separated on '+' characters.  If valid, return
        a list of the renderer names

        Parameters
        ----------
        renderers_string: str

        Returns
        -------
        list of str
        """
        if not isinstance(renderers_string, str):
            raise ValueError('Renderer must be specified as a string')
        renderer_names = renderers_string.split('+')
        invalid = [name for name in renderer_names if name not in self]
        if invalid:
            raise ValueError('\nInvalid named renderer(s) received: {}'.format(str(invalid)))
        return renderer_names

    def __repr__(self):
        return 'Renderers configuration\n-----------------------\n    Default renderer: {default}\n    Available renderers:\n{available}\n'.format(default=repr(self.default), available=self._available_renderers_str())

    def _available_renderers_str(self):
        """
        Return nicely wrapped string representation of all
        available renderer names
        """
        available = '\n'.join(textwrap.wrap(repr(list(self)), width=79 - 8, initial_indent=' ' * 8, subsequent_indent=' ' * 9))
        return available

    def _build_mime_bundle(self, fig_dict, renderers_string=None, **kwargs):
        """
        Build a mime bundle dict containing a kev/value pair for each
        MimetypeRenderer specified in either the default renderer string,
        or in the supplied renderers_string argument.

        Note that this method skips any renderers that are not subclasses
        of MimetypeRenderer.

        Parameters
        ----------
        fig_dict: dict
            Figure dictionary
        renderers_string: str or None (default None)
            Renderer string to process rather than the current default
            renderer string

        Returns
        -------
        dict
        """
        if renderers_string:
            renderer_names = self._validate_coerce_renderers(renderers_string)
            renderers_list = [self[name] for name in renderer_names]
            for renderer in renderers_list:
                if isinstance(renderer, MimetypeRenderer):
                    renderer.activate()
        else:
            self._activate_pending_renderers(cls=MimetypeRenderer)
            renderers_list = self._default_renderers
        bundle = {}
        for renderer in renderers_list:
            if isinstance(renderer, MimetypeRenderer):
                renderer = copy(renderer)
                for k, v in kwargs.items():
                    if hasattr(renderer, k):
                        setattr(renderer, k, v)
                bundle.update(renderer.to_mimebundle(fig_dict))
        return bundle

    def _perform_external_rendering(self, fig_dict, renderers_string=None, **kwargs):
        """
        Perform external rendering for each ExternalRenderer specified
        in either the default renderer string, or in the supplied
        renderers_string argument.

        Note that this method skips any renderers that are not subclasses
        of ExternalRenderer.

        Parameters
        ----------
        fig_dict: dict
            Figure dictionary
        renderers_string: str or None (default None)
            Renderer string to process rather than the current default
            renderer string

        Returns
        -------
        None
        """
        if renderers_string:
            renderer_names = self._validate_coerce_renderers(renderers_string)
            renderers_list = [self[name] for name in renderer_names]
            for renderer in renderers_list:
                if isinstance(renderer, ExternalRenderer):
                    renderer.activate()
        else:
            self._activate_pending_renderers(cls=ExternalRenderer)
            renderers_list = self._default_renderers
        for renderer in renderers_list:
            if isinstance(renderer, ExternalRenderer):
                renderer = copy(renderer)
                for k, v in kwargs.items():
                    if hasattr(renderer, k):
                        setattr(renderer, k, v)
                renderer.render(fig_dict)