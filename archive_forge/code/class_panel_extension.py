import ast
import copy
import importlib
import inspect
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from weakref import WeakKeyDictionary
import param
from bokeh.core.has_props import _default_resolver
from bokeh.document import Document
from bokeh.model import Model
from bokeh.settings import settings as bk_settings
from pyviz_comms import (
from .io.logging import panel_log_handler
from .io.state import state
from .util import param_watchers
class panel_extension(_pyviz_extension):
    """
    Initializes and configures Panel. You should always run `pn.extension`.
    This will

    - Initialize the `pyviz` notebook extension to enable bi-directional
    communication and for example plotting with Bokeh.
    - Load `.js` libraries (positional arguments).
    - Update the global configuration `pn.config`
    (keyword arguments).

    Parameters
    ----------
    *args : list[str]
        Positional arguments listing the extension to load. For example "plotly",
        "tabulator".
    **params : dict[str,Any]
        Keyword arguments to be set on the `pn.config` element. See
        https://panel.holoviz.org/api/config.html

    :Example:

    >>> import panel as pn
    >>> pn.extension("plotly", sizing_mode="stretch_width", template="fast")

    This will

    - Initialize the `pyviz` notebook extension.
    - Enable you to use the `Plotly` pane by loading `plotly.js`.
    - Set the default `sizing_mode` to `stretch_width` instead of `fixed`.
    - Set the global configuration `pn.config.template` to `fast`, i.e. you
    will be using the `FastListTemplate`.
    """
    _loaded = False
    _imports = {'ace': 'panel.models.ace', 'codeeditor': 'panel.models.ace', 'deckgl': 'panel.models.deckgl', 'echarts': 'panel.models.echarts', 'ipywidgets': 'panel.io.ipywidget', 'jsoneditor': 'panel.models.jsoneditor', 'katex': 'panel.models.katex', 'mathjax': 'panel.models.mathjax', 'perspective': 'panel.models.perspective', 'plotly': 'panel.models.plotly', 'tabulator': 'panel.models.tabulator', 'terminal': 'panel.models.terminal', 'texteditor': 'panel.models.quill', 'vizzu': 'panel.models.vizzu', 'vega': 'panel.models.vega', 'vtk': 'panel.models.vtk'}
    _globals = {'deckgl': ['deck'], 'echarts': ['echarts'], 'floatpanel': ['jsPanel'], 'gridstack': ['GridStack'], 'katex': ['katex'], 'mathjax': ['MathJax'], 'perspective': ["customElements.get('perspective-viewer')"], 'plotly': ['Plotly'], 'tabulator': ['Tabulator'], 'terminal': ['Terminal', 'xtermjs'], 'vega': ['vega'], 'vizzu': ['Vizzu'], 'vtk': ['vtk']}
    _loaded_extensions = []
    _comms_detected_before = False

    def __call__(self, *args, **params):
        from .reactive import ReactiveHTML, ReactiveHTMLMetaclass
        reactive_exts = {v._extension_name: v for k, v in param.concrete_descendents(ReactiveHTML).items()}
        newly_loaded = [arg for arg in args if arg not in panel_extension._loaded_extensions]
        if state.curdoc and state.curdoc not in state._extensions_:
            state._extensions_[state.curdoc] = []
        if params.get('ready_notification') or params.get('disconnect_notification'):
            params['notifications'] = True
        if params.get('notifications', config.notifications) and 'notifications' not in args:
            args += ('notifications',)
        for arg in args:
            if arg == 'notifications' and 'notifications' not in params:
                params['notifications'] = True
            if arg == 'ipywidgets':
                from .io.resources import CSS_URLS
                params['css_files'] = params.get('css_files', []) + [CSS_URLS['font-awesome']]
            if arg in self._imports:
                try:
                    if arg == 'ipywidgets' and get_ipython() and ('PANEL_IPYWIDGET' not in os.environ):
                        continue
                except Exception:
                    pass
                module = self._imports[arg]
                if module in sys.modules:
                    for model in sys.modules[module].__dict__.values():
                        if isinstance(model, type) and issubclass(model, Model):
                            qual = getattr(model, '__qualified_model__', None)
                            if qual and qual not in _default_resolver.known_models:
                                _default_resolver.add(model)
                else:
                    __import__(module)
                self._loaded_extensions.append(arg)
                if state.curdoc:
                    state._extensions_[state.curdoc].append(arg)
            elif arg in reactive_exts:
                if state.curdoc:
                    state._extensions.append(arg)
                ReactiveHTMLMetaclass._loaded_extensions.add(arg)
            else:
                self.param.warning('%s extension not recognized and will be skipped.' % arg)
        for k, v in params.items():
            if k == 'design' and isinstance(v, str):
                from .theme import Design
                try:
                    importlib.import_module(f'panel.theme.{self._design}')
                except Exception:
                    pass
                designs = {p.lower(): t for p, t in param.concrete_descendents(Design).items()}
                if v not in designs:
                    raise ValueError(f'Design {v!r} was not recognized, available design systems include: {list(designs)}.')
                setattr(config, k, designs[v])
            elif k in ('css_files', 'raw_css', 'global_css'):
                if not isinstance(v, list):
                    raise ValueError('%s should be supplied as a list, not as a %s type.' % (k, type(v).__name__))
                existing = getattr(config, k)
                existing.extend([new for new in v if new not in existing])
            elif k == 'js_files':
                getattr(config, k).update(v)
            else:
                setattr(config, k, v)
        if config.apply_signatures:
            self._apply_signatures()
        loaded = self._loaded
        panel_extension._loaded = True
        if loaded and args == ('vtk',) and ('vtk' in self._loaded_extensions):
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            if len(calframe) >= 3 and 'pyvista' in calframe[2].filename:
                return
        if 'holoviews' in sys.modules:
            import holoviews as hv
            import holoviews.plotting.bokeh
            loaded = loaded or getattr(hv.extension, '_loaded', False)
            if hv.Store.current_backend in hv.Store.renderers:
                backend = hv.Store.current_backend
            else:
                backend = 'bokeh'
            if not loaded or ((loaded and backend != hv.Store.current_backend) and hasattr(hv.Store, 'set_current_backend')):
                hv.Store.set_current_backend(backend)
            else:
                hv.Store.current_backend = backend
        if not loaded and config.load_entry_points:
            self._load_entry_points()
        try:
            ip = params.pop('ip', None) or get_ipython()
        except Exception:
            return
        from .io.notebook import load_notebook
        self._detect_comms(params)
        panel_extension._loaded_extensions += newly_loaded
        if hasattr(ip, 'kernel') and (not loaded) and (not config._doc_build):
            _JupyterCommManager.get_client_comm(self._process_comm_msg, 'hv-extension-comm')
            state._comm_manager = _JupyterCommManager
        if 'ipywidgets' in sys.modules and config.embed:
            __import__(self._imports['ipywidgets'])
        nb_loaded = published = getattr(self, '_repeat_execution_in_cell', False)
        if 'holoviews' in sys.modules:
            if getattr(hv.extension, '_loaded', False):
                nb_loaded = True
            else:
                with param.logging_level('ERROR'):
                    hv.plotting.Renderer.load_nb(config.inline)
                    if hasattr(hv.plotting.Renderer, '_render_with_panel'):
                        nb_loaded = True
        bk_settings.simple_ids.set_value(False)
        if hasattr(ip, 'kernel'):
            load_notebook(config.inline, reloading=nb_loaded)
        if not published:
            self._display_globals()

    @staticmethod
    def _display_globals():
        if config.browser_info and state.browser_info:
            doc = Document()
            comm = state._comm_manager.get_server_comm()
            model = state.browser_info._render_model(doc, comm)
            bundle, meta = state.browser_info._render_mimebundle(model, doc, comm)
            display(bundle, metadata=meta, raw=True)
        if config.notifications:
            display(state.notifications)

    def _detect_comms(self, params):
        called_before = self._comms_detected_before
        self._comms_detected_before = True
        if 'comms' in params:
            config.comms = params.pop('comms')
            return
        if called_before:
            return
        if 'google.colab' in sys.modules:
            try:
                import jupyter_bokeh
                config.comms = 'colab'
            except Exception:
                warnings.warn('Using Panel interactively in Colab notebooks requires the jupyter_bokeh package to be installed. Install it with:\n\n    !pip install jupyter_bokeh\n\nand try again.', stacklevel=5)
            return
        if 'VSCODE_CWD' in os.environ or 'VSCODE_PID' in os.environ:
            try:
                import jupyter_bokeh
                config.comms = 'vscode'
            except Exception:
                warnings.warn('Using Panel interactively in VSCode notebooks requires the jupyter_bokeh package to be installed. You can install it with:\n\n   pip install jupyter_bokeh\n\nor:\n    conda install jupyter_bokeh\n\nand try again.', stacklevel=5)
            self._ignore_bokeh_warnings()
            return

    def _apply_signatures(self):
        from inspect import Parameter, Signature
        from .viewable import Viewable
        descendants = param.concrete_descendents(Viewable)
        for cls in reversed(list(descendants.values())):
            if cls.__doc__ is None:
                pass
            elif cls.__doc__.startswith('params'):
                prefix = cls.__doc__.split('\n')[0]
                cls.__doc__ = cls.__doc__.replace(prefix, '')
            sig = inspect.signature(cls.__init__)
            sig_params = list(sig.parameters.values())
            if not sig_params or sig_params[-1] != Parameter('params', Parameter.VAR_KEYWORD):
                continue
            parameters = sig_params[:-1]
            processed_kws, keyword_groups = (set(), [])
            for scls in reversed(cls.mro()):
                keyword_group = []
                for k, v in sorted(scls.__dict__.items()):
                    if isinstance(v, param.Parameter) and k not in processed_kws and (not v.readonly):
                        keyword_group.append(k)
                        processed_kws.add(k)
                keyword_groups.append(keyword_group)
            parameters += [Parameter(name, Parameter.KEYWORD_ONLY) for kws in reversed(keyword_groups) for name in kws if name not in sig.parameters]
            kwarg_name = '_kwargs' if 'kwargs' in processed_kws else 'kwargs'
            parameters.append(Parameter(kwarg_name, Parameter.VAR_KEYWORD))
            cls.__init__.__signature__ = Signature(parameters, return_annotation=sig.return_annotation)

    def _load_entry_points(self):
        """
        Load entry points from external packages.
        Import is performed here, so any importlib
        can be easily bypassed by switching off the configuration flag.
        Also, there is no reason to waste time importing this module
        if it won't be used.
        """
        from .entry_points import load_entry_points
        load_entry_points('panel.extension')

    def _ignore_bokeh_warnings(self):
        from bokeh.util.warnings import BokehUserWarning
        warnings.filterwarnings('ignore', category=BokehUserWarning, message='reference already known')