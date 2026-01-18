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
class _config(_base_config):
    """
    Holds global configuration options for Panel.

    The options can be set directly on the global config instance, via
    keyword arguments in the extension or via environment variables.

    For example to set the embed option the following approaches can be used:

        pn.config.embed = True

        pn.extension(embed=True)

        os.environ['PANEL_EMBED'] = 'True'

    Reference: Currently none

    :Example:

    >>> pn.config.loading_spinner = 'bar'
    """
    admin_plugins = param.List(default=[], item_type=tuple, doc='\n        A list of tuples containing a title and a function that returns\n        an additional panel to be rendered into the admin page.')
    apply_signatures = param.Boolean(default=True, doc='\n        Whether to set custom Signature which allows tab-completion\n        in some IDEs and environments.')
    authorize_callback = param.Callable(default=None, doc='\n        Authorization callback that is invoked when authentication\n        is enabled. The callback is given the user information returned\n        by the configured Auth provider and should return True or False\n        depending on whether the user is authorized to access the\n        application. The callback may also contain a second parameter,\n        which is the requested path the user is making. If the user\n        is authenticated and has explicit access to the path, then\n        the callback should return True otherwise it should return\n        False.')
    auth_template = param.Path(default=None, doc='\n        A jinja2 template rendered when the authorize_callback determines\n        that a user in not authorized to access the application.')
    autoreload = param.Boolean(default=False, doc='\n        Whether to autoreload server when script changes.')
    basic_auth_template = param.Path(default=None, doc='\n        A jinja2 template to override the default Basic Authentication\n        login page.')
    browser_info = param.Boolean(default=True, doc='\n        Whether to request browser info from the frontend.')
    defer_load = param.Boolean(default=False, doc='\n        Whether to defer load of rendered functions.')
    design = param.ClassSelector(class_=None, is_instance=False, doc='\n        The design system to use to style components.')
    disconnect_notification = param.String(doc='\n        The notification to display to the user when the connection\n        to the server is dropped.')
    exception_handler = param.Callable(default=None, doc='\n        General exception handler for events.')
    global_css = param.List(default=[], doc='\n        List of raw CSS to be added to the header.')
    global_loading_spinner = param.Boolean(default=False, doc='\n        Whether to add a global loading spinner for the whole application.')
    layout_compatibility = param.Selector(default='warn', objects=['warn', 'error'], doc='\n        Provide compatibility for older layout specifications. Incompatible\n        specifications will trigger warnings by default but can be set to error.\n        Compatibility to be set to error by default in Panel 1.1.')
    load_entry_points = param.Boolean(default=True, doc='\n        Load entry points from external packages.')
    loading_indicator = param.Boolean(default=False, doc='\n        Whether a loading indicator is shown by default while panes are updating.')
    loading_spinner = param.Selector(default='arc', objects=['arc', 'arcs', 'bar', 'dots', 'petal'], doc='\n        Loading indicator to use when component loading parameter is set.')
    loading_color = param.Color(default='#c3c3c3', doc='\n        Color of the loading indicator.')
    loading_max_height = param.Integer(default=400, doc='\n        Maximum height of the loading indicator.')
    notifications = param.Boolean(default=False, doc='\n        Whether to enable notifications functionality.')
    profiler = param.Selector(default=None, allow_None=True, objects=['pyinstrument', 'snakeviz', 'memray'], doc='\n        The profiler engine to enable.')
    ready_notification = param.String(doc='\n        The notification to display when the application is ready and\n        fully loaded.')
    reuse_sessions = param.Boolean(default=False, doc='\n        Whether to reuse a session for the initial request to speed up\n        the initial page render. Note that if the initial page differs\n        between sessions, e.g. because it uses query parameters to modify\n        the rendered content, then this option will result in the wrong\n        content being rendered. Define a session_key_func to ensure that\n        reused sessions are only reused when appropriate.')
    session_key_func = param.Callable(default=None, doc='\n        Used in conjunction with the reuse_sessions option, the\n        session_key_func is given a tornado.httputil.HTTPServerRequest\n        and should return a key that uniquely captures a session.')
    safe_embed = param.Boolean(default=False, doc='\n        Ensure all bokeh property changes trigger events which are\n        embedded. Useful when only partial updates are made in an\n        app, e.g. when working with HoloViews.')
    session_history = param.Integer(default=0, bounds=(-1, None), doc='\n        If set to a non-negative value this determines the maximum length\n        of the pn.state.session_info dictionary, which tracks\n        information about user sessions. A value of -1 indicates an\n        unlimited history.')
    sizing_mode = param.ObjectSelector(default=None, objects=['fixed', 'stretch_width', 'stretch_height', 'stretch_both', 'scale_width', 'scale_height', 'scale_both', None], doc='\n        Specify the default sizing mode behavior of panels.')
    template = param.ObjectSelector(default=None, doc='\n        The default template to render served applications into.')
    throttled = param.Boolean(default=False, doc='\n        If sliders and inputs should be throttled until release of mouse.')
    _admin = param.Boolean(default=False, doc='Whether the admin panel is enabled.')
    _admin_endpoint = param.String(default=None, doc='Name to use for the admin endpoint.')
    _admin_log_level = param.Selector(default='DEBUG', objects=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], doc='Log level of the Admin Panel logger')
    _comms = param.ObjectSelector(default='default', objects=['default', 'ipywidgets', 'vscode', 'colab'], doc='\n        Whether to render output in Jupyter with the default Jupyter\n        extension or use the jupyter_bokeh ipywidget model.')
    _console_output = param.ObjectSelector(default='accumulate', allow_None=True, objects=['accumulate', 'replace', 'disable', False], doc='\n        How to log errors and stdout output triggered by callbacks\n        from Javascript in the notebook.')
    _cookie_secret = param.String(default=None, doc='\n        Configure to enable getting/setting secure cookies.')
    _embed = param.Boolean(default=False, allow_None=True, doc='\n        Whether plot data will be embedded.')
    _embed_json = param.Boolean(default=False, doc='\n        Whether to save embedded state to json files.')
    _embed_json_prefix = param.String(default='', doc='\n        Prefix for randomly generated json directories.')
    _embed_load_path = param.String(default=None, doc='\n        Where to load json files for embedded state.')
    _embed_save_path = param.String(default='./', doc='\n        Where to save json files for embedded state.')
    _log_level = param.Selector(default='WARNING', objects=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], doc='Log level of Panel loggers')
    _npm_cdn = param.Selector(default='https://cdn.jsdelivr.net/npm', objects=['https://unpkg.com', 'https://cdn.jsdelivr.net/npm'], doc='\n        The CDN to load NPM packages from if resources are served from\n        CDN. Allows switching between [https://unpkg.com](https://unpkg.com) and\n        [https://cdn.jsdelivr.net/npm](https://cdn.jsdelivr.net/npm) for most resources.')
    _nthreads = param.Integer(default=None, doc='\n        When set to a non-None value a thread pool will be started.\n        Whenever an event arrives from the frontend it will be\n        dispatched to the thread pool to be processed.')
    _basic_auth = param.ClassSelector(default=None, class_=(dict, str), allow_None=True, doc='\n        Password, dictionary with a mapping from username to password\n        or filepath containing JSON to use with the basic auth\n        provider.')
    _oauth_provider = param.ObjectSelector(default=None, allow_None=True, objects=[], doc='\n        Select between a list of authentication providers.')
    _oauth_expiry = param.Number(default=1, bounds=(0, None), doc='\n        Expiry of the OAuth cookie in number of days.')
    _oauth_key = param.String(default=None, doc='\n        A client key to provide to the OAuth provider.')
    _oauth_secret = param.String(default=None, doc='\n        A client secret to provide to the OAuth provider.')
    _oauth_jwt_user = param.String(default=None, doc='\n        The key in the ID JWT token to consider the user.')
    _oauth_redirect_uri = param.String(default=None, doc='\n        A redirect URI to provide to the OAuth provider.')
    _oauth_encryption_key = param.ClassSelector(default=None, class_=bytes, doc='\n        A random string used to encode OAuth related user information.')
    _oauth_extra_params = param.Dict(default={}, doc='\n        Additional parameters required for OAuth provider.')
    _oauth_guest_endpoints = param.List(default=None, doc='\n        List of endpoints that can be accessed as a guest without authenticating.')
    _oauth_optional = param.Boolean(default=False, doc='\n        Whether the user will be forced to go through login flow or if\n        they can access all applications as a guest.')
    _oauth_refresh_tokens = param.Boolean(default=False, doc='\n        Whether to automatically refresh access tokens in the background.')
    _inline = param.Boolean(default=_LOCAL_DEV_VERSION, allow_None=True, doc='\n        Whether to inline JS and CSS resources. If disabled, resources\n        are loaded from CDN if one is available.')
    _theme = param.ObjectSelector(default=None, objects=['default', 'dark'], allow_None=True, doc='\n        The theme to apply to components.')
    _globals = {'admin_plugins', 'autoreload', 'comms', 'cookie_secret', 'nthreads', 'oauth_provider', 'oauth_expiry', 'oauth_key', 'oauth_secret', 'oauth_jwt_user', 'oauth_redirect_uri', 'oauth_encryption_key', 'oauth_extra_params', 'npm_cdn', 'layout_compatibility', 'oauth_refresh_tokens', 'oauth_guest_endpoints', 'oauth_optional', 'admin'}
    _truthy = ['True', 'true', '1', True, 1]
    _session_config = WeakKeyDictionary()

    def __init__(self, **params):
        super().__init__(**params)
        self._validating = False
        for p in self._parameter_set:
            if p.startswith('_') and p[1:] not in _config._globals:
                setattr(self, p + '_', None)
        if self.log_level:
            panel_log_handler.setLevel(self.log_level)

    @param.depends('_nthreads', watch=True, on_init=True)
    def _set_thread_pool(self):
        if self.nthreads is None:
            if state._thread_pool is not None:
                state._thread_pool.shutdown(wait=False)
            state._thread_pool = None
            return
        if state._thread_pool:
            raise RuntimeError('Thread pool already running')
        threads = self.nthreads if self.nthreads else None
        state._thread_pool = ThreadPoolExecutor(max_workers=threads)

    @param.depends('notifications', watch=True)
    def _setup_notifications(self):
        from .io.notifications import NotificationArea
        from .reactive import ReactiveHTMLMetaclass
        if self.notifications and 'notifications' not in ReactiveHTMLMetaclass._loaded_extensions:
            ReactiveHTMLMetaclass._loaded_extensions.add('notifications')
        if not state.curdoc:
            state._notification = NotificationArea()

    @param.depends('disconnect_notification', 'ready_notification', watch=True)
    def _enable_notifications(self):
        if self.disconnect_notification or self.ready_notification:
            self.notifications = True

    @contextmanager
    def set(self, **kwargs):
        values = [(k, v) for k, v in self.param.values().items() if k != 'name']
        overrides = [(k, getattr(self, k + '_')) for k in _config._parameter_set if k.startswith('_') and k[1:] not in _config._globals]
        for k, v in kwargs.items():
            setattr(self, k, v)
        try:
            yield
        finally:
            self.param.update(**dict(values))
            for k, v in overrides:
                setattr(self, k + '_', v)

    def __setattr__(self, attr, value):
        from .io.state import state
        if hasattr(self, '_param__private'):
            init = getattr(self._param__private, 'initialized', False)
        else:
            init = getattr(self, 'initialized', False)
        if not init or (attr.startswith('_') and attr.endswith('_')) or attr == '_validating':
            return super().__setattr__(attr, value)
        value = getattr(self, f'_{attr}_hook', lambda x: x)(value)
        if attr in _config._globals or (attr.startswith('_') and attr[1:] in _config._globals) or self.param._TRIGGER:
            super().__setattr__(attr if attr in self.param else f'_{attr}', value)
        elif state.curdoc is not None:
            if attr in _config._parameter_set:
                validate_config(self, attr, value)
            elif f'_{attr}' in _config._parameter_set:
                validate_config(self, f'_{attr}', value)
            else:
                raise AttributeError(f'{attr!r} is not a valid config parameter.')
            if state.curdoc not in self._session_config:
                self._session_config[state.curdoc] = {}
            self._session_config[state.curdoc][attr] = value
            watchers = param_watchers(self).get(attr, {}).get('value', [])
            for w in watchers:
                w.fn()
        elif f'_{attr}' in _config._parameter_set and hasattr(self, f'_{attr}_'):
            validate_config(self, f'_{attr}', value)
            super().__setattr__(f'_{attr}_', value)
        else:
            super().__setattr__(attr, value)

    @param.depends('_log_level', watch=True)
    def _update_log_level(self):
        panel_log_handler.setLevel(self._log_level)

    @param.depends('_admin_log_level', watch=True)
    def _update_admin_log_level(self):
        from .io.admin import log_handler as admin_log_handler
        admin_log_handler.setLevel(self._log_level)

    def __getattribute__(self, attr):
        """
        Ensures that configuration parameters that are defined per
        session are stored in a per-session dictionary. This is to
        ensure that even on first access mutable parameters do not
        end up being modified.
        """
        if attr in ('_param__private', '_globals', '_parameter_set', '__class__', 'param'):
            return super().__getattribute__(attr)
        from .io.state import state
        session_config = super().__getattribute__('_session_config')
        curdoc = state.curdoc
        if curdoc and curdoc not in session_config:
            session_config[curdoc] = {}
        if attr in ('raw_css', 'global_css', 'css_files', 'js_files', 'js_modules') and curdoc and (attr not in session_config[curdoc]):
            new_obj = copy.copy(super().__getattribute__(attr))
            setattr(self, attr, new_obj)
        if attr in _config._globals or attr == 'theme':
            return super().__getattribute__(attr)
        elif curdoc and curdoc in session_config and (attr in session_config[curdoc]):
            return session_config[curdoc][attr]
        elif f'_{attr}' in _config._parameter_set and getattr(self, f'_{attr}_') is not None:
            return super().__getattribute__(f'_{attr}_')
        return super().__getattribute__(attr)

    def _console_output_hook(self, value):
        return value if value else 'disable'

    def _template_hook(self, value):
        if isinstance(value, str):
            return self.param.template.names[value]
        return value

    @property
    def _doc_build(self):
        return os.environ.get('PANEL_DOC_BUILD')

    @property
    def admin(self):
        return self._admin

    @property
    def admin_endpoint(self):
        return os.environ.get('PANEL_ADMIN_ENDPOINT', self._admin_endpoint)

    @property
    def admin_log_level(self):
        admin_log_level = os.environ.get('PANEL_ADMIN_LOG_LEVEL', self._admin_log_level)
        return admin_log_level.upper() if admin_log_level else None

    @property
    def console_output(self):
        if self._doc_build:
            return 'disable'
        else:
            return os.environ.get('PANEL_CONSOLE_OUTPUT', _config._console_output)

    @property
    def embed(self):
        return os.environ.get('PANEL_EMBED', _config._embed) in self._truthy

    @property
    def comms(self):
        return os.environ.get('PANEL_COMMS', self._comms)

    @property
    def embed_json(self):
        return os.environ.get('PANEL_EMBED_JSON', _config._embed_json) in self._truthy

    @property
    def embed_json_prefix(self):
        return os.environ.get('PANEL_EMBED_JSON_PREFIX', _config._embed_json_prefix)

    @property
    def embed_save_path(self):
        return os.environ.get('PANEL_EMBED_SAVE_PATH', _config._embed_save_path)

    @property
    def embed_load_path(self):
        return os.environ.get('PANEL_EMBED_LOAD_PATH', _config._embed_load_path)

    @property
    def inline(self):
        return os.environ.get('PANEL_INLINE', _config._inline) in self._truthy

    @property
    def log_level(self):
        log_level = os.environ.get('PANEL_LOG_LEVEL', self._log_level)
        return log_level.upper() if log_level else None

    @property
    def npm_cdn(self):
        return os.environ.get('PANEL_NPM_CDN', _config._npm_cdn)

    @property
    def nthreads(self):
        nthreads = os.environ.get('PANEL_NUM_THREADS', self._nthreads)
        return None if nthreads is None else int(nthreads)

    @property
    def basic_auth(self):
        provider = os.environ.get('PANEL_BASIC_AUTH', self._oauth_provider)
        return provider.lower() if provider else None

    @property
    def oauth_provider(self):
        provider = os.environ.get('PANEL_OAUTH_PROVIDER', self._oauth_provider)
        return provider.lower() if provider else None

    @property
    def oauth_expiry(self):
        provider = os.environ.get('PANEL_OAUTH_EXPIRY', self._oauth_expiry)
        return float(provider)

    @property
    def oauth_key(self):
        return os.environ.get('PANEL_OAUTH_KEY', self._oauth_key)

    @property
    def cookie_secret(self):
        return os.environ.get('PANEL_COOKIE_SECRET', os.environ.get('BOKEH_COOKIE_SECRET', self._cookie_secret))

    @property
    def oauth_secret(self):
        return os.environ.get('PANEL_OAUTH_SECRET', self._oauth_secret)

    @property
    def oauth_redirect_uri(self):
        return os.environ.get('PANEL_OAUTH_REDIRECT_URI', self._oauth_redirect_uri)

    @property
    def oauth_jwt_user(self):
        return os.environ.get('PANEL_OAUTH_JWT_USER', self._oauth_jwt_user)

    @property
    def oauth_refresh_tokens(self):
        refresh = os.environ.get('PANEL_OAUTH_REFRESH_TOKENS', self._oauth_refresh_tokens)
        if isinstance(refresh, bool):
            return refresh
        return refresh.lower() in ('1', 'true')

    @property
    def oauth_encryption_key(self):
        return os.environ.get('PANEL_OAUTH_ENCRYPTION', self._oauth_encryption_key)

    @property
    def oauth_extra_params(self):
        if 'PANEL_OAUTH_EXTRA_PARAMS' in os.environ:
            return ast.literal_eval(os.environ['PANEL_OAUTH_EXTRA_PARAMS'])
        else:
            return self._oauth_extra_params

    @property
    def oauth_guest_endpoints(self):
        if 'PANEL_OAUTH_GUEST_ENDPOINTS' in os.environ:
            return ast.literal_eval(os.environ['PANEL_OAUTH_GUEST_ENDPOINTS'])
        else:
            return self._oauth_guest_endpoints

    @property
    def oauth_optional(self):
        optional = os.environ.get('PANEL_OAUTH_OPTIONAL', self._oauth_optional)
        if isinstance(optional, bool):
            return optional
        return optional.lower() in ('1', 'true')

    @property
    def theme(self):
        curdoc = state.curdoc
        if curdoc and 'theme' in self._session_config.get(curdoc, {}):
            return self._session_config[curdoc]['theme']
        elif self._theme_:
            return self._theme_
        elif isinstance(state.session_args, dict) and state.session_args:
            theme = state.session_args.get('theme', [b'default'])[0].decode('utf-8')
            if theme in self.param._theme.objects:
                return theme
        return 'default'