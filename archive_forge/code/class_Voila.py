import gettext
import io
import sys
import json
import logging
import threading
import tempfile
import os
import shutil
import signal
import socket
import webbrowser
import errno
import random
import jinja2
import tornado.ioloop
import tornado.web
from traitlets.config.application import Application
from traitlets.config.loader import Config
from traitlets import Unicode, Integer, Bool, Dict, List, Callable, default, Type, Bytes
from jupyter_server.services.contents.largefilemanager import LargeFileManager
from jupyter_server.services.kernels.handlers import KernelHandler
from jupyter_server.base.handlers import path_regex, FileFindHandler
from jupyter_server.config_manager import recursive_update
from jupyterlab_server.themes_handler import ThemesHandler
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_core.paths import jupyter_config_path, jupyter_path
from .paths import ROOT, STATIC_ROOT, collect_template_paths, collect_static_paths
from .handler import VoilaHandler
from .treehandler import VoilaTreeHandler
from ._version import __version__
from .static_file_handler import MultiStaticFileHandler, TemplateStaticFileHandler, WhiteListFileHandler
from .configuration import VoilaConfiguration
from .execute import VoilaExecutor
from .exporter import VoilaExporter
from .shutdown_kernel_handler import VoilaShutdownKernelHandler
from .voila_kernel_manager import voila_kernel_manager_factory
from .request_info_handler import RequestInfoSocketHandler
from .utils import create_include_assets_functions
class Voila(Application):
    name = 'voila'
    version = __version__
    examples = 'voila example.ipynb --port 8888'
    flags = {'debug': ({'Voila': {'log_level': logging.DEBUG}, 'VoilaConfiguration': {'show_tracebacks': True}}, _('Set the log level to logging.DEBUG, and show exception tracebacks in output.')), 'no-browser': ({'Voila': {'open_browser': False}}, _("Don't open the notebook in a browser after startup.")), 'show-margins': ({'VoilaConfiguration': {'show_margins': True}}, _('Show left and right margins for the "lab" template, this gives a "classic" template look'))}
    if JUPYTER_SERVER_2:
        flags = {**flags, 'token': ({'Voila': {'auto_token': True}}, _(''))}
    description = Unicode('voila [OPTIONS] NOTEBOOK_FILENAME\n\n        This launches a stand-alone server for read-only notebooks.\n        ')
    option_description = Unicode('\n        notebook_path:\n            File name of the Jupyter notebook to display.\n        ')
    notebook_filename = Unicode()
    port = Integer(8866, config=True, help=_('Port of the Voilà server. Default 8866.'))
    autoreload = Bool(False, config=True, help=_('Will autoreload to server and the page when a template, js file or Python code changes'))
    root_dir = Unicode(config=True, help=_('The directory to use for notebooks.'))
    static_root = Unicode(STATIC_ROOT, config=True, help=_('Directory holding static assets (HTML, JS and CSS files).'))
    aliases = {'autoreload': 'Voila.autoreload', 'base_url': 'Voila.base_url', 'port': 'Voila.port', 'static': 'Voila.static_root', 'server_url': 'Voila.server_url', 'pool_size': 'VoilaConfiguration.default_pool_size', 'enable_nbextensions': 'VoilaConfiguration.enable_nbextensions', 'nbextensions_path': 'VoilaConfiguration.nbextensions_path', 'show_tracebacks': 'VoilaConfiguration.show_tracebacks', 'preheat_kernel': 'VoilaConfiguration.preheat_kernel', 'strip_sources': 'VoilaConfiguration.strip_sources', 'template': 'VoilaConfiguration.template', 'theme': 'VoilaConfiguration.theme'}
    if JUPYTER_SERVER_2:
        aliases = {**aliases, 'token': 'Voila.token'}
    classes = [VoilaConfiguration, VoilaExecutor, VoilaExporter]
    connection_dir_root = Unicode(config=True, help=_('Location of temporary connection files. Defaults to system `tempfile.gettempdir()` value.'))
    connection_dir = Unicode()
    base_url = Unicode('/', config=True, help=_('Path for Voilà API calls. If server_url is unset, this will be             used for both the base route of the server and the client.             If server_url is set, the server will server the routes prefixed             by server_url, while the client will prefix by base_url (this is             useful in reverse proxies).'))
    server_url = Unicode(None, config=True, allow_none=True, help=_('Path to prefix to Voilà API handlers. Leave unset to default to base_url'))
    notebook_path = Unicode(None, config=True, allow_none=True, help=_('path to notebook to serve with Voilà'))
    template_paths = List([], config=True, help=_('path to jinja2 templates'))
    static_paths = List([STATIC_ROOT], config=True, help=_('paths to static assets'))
    port_retries = Integer(50, config=True, help=_('The number of additional ports to try if the specified port is not available.'))
    ip = Unicode('localhost', config=True, help=_('The IP address the notebook server will listen on.'))
    open_browser = Bool(True, config=True, help=_('Whether to open in a browser after starting.\n                        The specific browser used is platform dependent and\n                        determined by the python standard library `webbrowser`\n                        module, unless it is overridden using the --browser\n                        (NotebookApp.browser) configuration option.\n                        '))
    browser = Unicode(u'', config=True, help='Specify what command to use to invoke a web\n                      browser when opening the notebook. If not specified, the\n                      default browser will be determined by the `webbrowser`\n                      standard library module, which allows setting of the\n                      BROWSER environment variable to override it.\n                      ')
    webbrowser_open_new = Integer(2, config=True, help=_('Specify Where to open the notebook on startup. This is the\n                                  `new` argument passed to the standard library method `webbrowser.open`.\n                                  The behaviour is not guaranteed, but depends on browser support. Valid\n                                  values are:\n                                  - 2 opens a new tab,\n                                  - 1 opens a new window,\n                                  - 0 opens in an existing window.\n                                  See the `webbrowser.open` documentation for details.\n                                  '))
    custom_display_url = Unicode(u'', config=True, help=_('Override URL shown to users.\n                                 Replace actual URL, including protocol, address, port and base URL,\n                                 with the given value when displaying URL to the users. Do not change\n                                 the actual connection URL. If authentication token is enabled, the\n                                 token is added to the custom URL automatically.\n                                 This option is intended to be used when the URL to display to the user\n                                 cannot be determined reliably by the Jupyter notebook server (proxified\n                                 or containerized setups for example).'))
    prelaunch_hook = Callable(default_value=None, allow_none=True, config=True, help=_('A function that is called prior to the launch of a new kernel instance\n            when a user visits the voila webpage. Used for custom user authorization\n            or any other necessary pre-launch functions.\n\n            Should be of the form:\n\n            def hook(req: tornado.web.RequestHandler,\n                    notebook: nbformat.NotebookNode,\n                    cwd: str)\n\n            Although most customizations can leverage templates, if you need access\n            to the request object (e.g. to inspect cookies for authentication),\n            or to modify the notebook itself (e.g. to inject some custom structure,\n            although much of this can be done by interacting with the kernel\n            in javascript) the prelaunch hook lets you do that.\n            '))
    if JUPYTER_SERVER_2:
        cookie_secret = Bytes(b'', config=True, help='The random bytes used to secure cookies.\n            By default this is a new random number every time you start the server.\n            Set it to a value in a config file to enable logins to persist across server sessions.\n\n            Note: Cookie secrets should be kept private, do not share config files with\n            cookie_secret stored in plaintext (you can read the value from a file).\n            ')
        token = Unicode(None, help='Token for identity provider ', allow_none=True).tag(config=True)
        auto_token = Bool(False, help='Generate token automatically ', allow_none=True).tag(config=True)

        @default('cookie_secret')
        def _default_cookie_secret(self):
            return os.urandom(32)
        authorizer_class = Type(default_value=AllowAllAuthorizer, klass=Authorizer, config=True, help=_('The authorizer class to use.'))
        identity_provider_class = Type(default_value=PasswordIdentityProvider, klass=IdentityProvider, config=True, help=_('The identity provider class to use.'))
        kernel_websocket_connection_class = Type(default_value=ZMQChannelsWebsocketConnection, klass=BaseKernelWebsocketConnection, config=True, help=_('The kernel websocket connection class to use.'))

    @property
    def display_url(self):
        if self.custom_display_url:
            url = self.custom_display_url
            if not url.endswith('/'):
                url += '/'
        else:
            if self.ip in ('', '0.0.0.0'):
                ip = '%s' % socket.gethostname()
            else:
                ip = self.ip
            url = self._url(ip)
        if JUPYTER_SERVER_2 and self.identity_provider.token:
            token = self.identity_provider.token if self.identity_provider.token_generated else '...'
            query = f'?token={token}'
        else:
            query = ''
        return f'{url}{query}'

    @property
    def connection_url(self):
        ip = self.ip if self.ip else 'localhost'
        return self._url(ip)

    def _url(self, ip):
        proto = 'http'
        return '%s://%s:%i%s' % (proto, ip, self.port, self.base_url)
    config_file_paths = List(Unicode(), config=True, help=_('Paths to search for voila.(py|json)'))
    tornado_settings = Dict({}, config=True, help=_('Extra settings to apply to tornado application, e.g. headers, ssl, etc'))

    @default('config_file_paths')
    def _config_file_paths_default(self):
        return [os.getcwd()] + jupyter_config_path()

    @default('connection_dir_root')
    def _default_connection_dir(self):
        connection_dir = tempfile.gettempdir()
        self.log.info('Using %s to store connection files' % connection_dir)
        return connection_dir

    @default('log_level')
    def _default_log_level(self):
        return logging.INFO

    @property
    def nbextensions_path(self):
        """The path to look for Javascript notebook extensions"""
        if self.voila_configuration.nbextensions_path:
            return self.voila_configuration.nbextensions_path
        path = jupyter_path('nbextensions')
        try:
            from IPython.paths import get_ipython_dir
        except ImportError:
            pass
        else:
            path.append(os.path.join(get_ipython_dir(), 'nbextensions'))
        return path

    @default('root_dir')
    def _default_root_dir(self):
        if self.notebook_path:
            return os.path.dirname(os.path.abspath(self.notebook_path))
        else:
            return os.getcwd()

    def _init_asyncio_patch(self):
        """set default asyncio policy to be compatible with tornado
        Tornado 6 (at least) is not compatible with the default
        asyncio implementation on Windows
        Pick the older SelectorEventLoopPolicy on Windows
        if the known-incompatible default policy is in use.
        do this as early as possible to make it a low priority and overridable
        ref: https://github.com/tornadoweb/tornado/issues/2608
        FIXME: if/when tornado supports the defaults in asyncio,
               remove and bump tornado requirement for py38
        """
        if sys.platform.startswith('win') and sys.version_info >= (3, 8):
            import asyncio
            try:
                from asyncio import WindowsProactorEventLoopPolicy, WindowsSelectorEventLoopPolicy
            except ImportError:
                pass
            else:
                if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
                    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

    def initialize(self, argv=None):
        self._init_asyncio_patch()
        self.log.debug('Searching path %s for config files', self.config_file_paths)
        super(Voila, self).initialize(argv)
        if len(self.extra_args) == 1:
            arg = self.extra_args[0]
            if not self.notebook_path:
                if os.path.isdir(arg):
                    self.root_dir = arg
                elif os.path.isfile(arg):
                    self.notebook_path = arg
                else:
                    raise ValueError('argument is neither a file nor a directory: %r' % arg)
        elif len(self.extra_args) != 0:
            raise ValueError('provided more than 1 argument: %r' % self.extra_args)
        self.load_config_file('voila', path=self.config_file_paths)
        self.voila_configuration = VoilaConfiguration(parent=self)
        self.setup_template_dirs()
        signal.signal(signal.SIGTERM, self._handle_signal_stop)

    def setup_template_dirs(self):
        if self.voila_configuration.template:
            template_name = self.voila_configuration.template
            self.template_paths = collect_template_paths(['voila', 'nbconvert'], template_name, prune=True)
            self.static_paths = collect_static_paths(['voila', 'nbconvert'], template_name)
            if JUPYTER_SERVER_2:
                self.static_paths.append(DEFAULT_STATIC_FILES_PATH)
            conf_paths = [os.path.join(d, 'conf.json') for d in self.template_paths]
            for p in conf_paths:
                if os.path.exists(p):
                    with open(p) as json_file:
                        conf = json.load(json_file)
                    if 'traitlet_configuration' in conf:
                        recursive_update(conf['traitlet_configuration'], self.voila_configuration.config.VoilaConfiguration)
                        self.voila_configuration.config.VoilaConfiguration = Config(conf['traitlet_configuration'])
        self.log.debug('using template: %s', self.voila_configuration.template)
        self.log.debug('template paths:\n\t%s', '\n\t'.join(self.template_paths))
        self.log.debug('static paths:\n\t%s', '\n\t'.join(self.static_paths))
        if self.notebook_path and (not os.path.exists(self.notebook_path)):
            raise ValueError('Notebook not found: %s' % self.notebook_path)

    def init_settings(self) -> Dict:
        """Initialize settings for Voila application."""
        self.server_url = self.server_url or self.base_url
        self.kernel_spec_manager = KernelSpecManager(parent=self)
        read_config_path = [os.path.join(p, 'serverconfig') for p in jupyter_config_path()]
        read_config_path += [os.path.join(p, 'nbconfig') for p in jupyter_config_path()]
        self.config_manager = ConfigManager(parent=self, read_config_path=read_config_path)
        self.contents_manager = LargeFileManager(parent=self)
        preheat_kernel: bool = self.voila_configuration.preheat_kernel
        pool_size: int = self.voila_configuration.default_pool_size
        if preheat_kernel and self.prelaunch_hook:
            raise Exception('`preheat_kernel` and `prelaunch_hook` are incompatible')
        kernel_manager_class = voila_kernel_manager_factory(self.voila_configuration.multi_kernel_manager_class, preheat_kernel, pool_size)
        self.kernel_manager = kernel_manager_class(parent=self, connection_dir=self.connection_dir, kernel_spec_manager=self.kernel_spec_manager, allowed_message_types=['comm_open', 'comm_close', 'comm_msg', 'comm_info_request', 'kernel_info_request', 'shutdown_request'])
        jenv_opt = {'autoescape': True}
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_paths), extensions=['jinja2.ext.i18n'], **jenv_opt)
        nbui = gettext.translation('nbui', localedir=os.path.join(ROOT, 'i18n'), fallback=True)
        env.install_gettext_translations(nbui, newstyle=False)
        if JUPYTER_SERVER_2:
            server_env = jinja2.Environment(loader=jinja2.FileSystemLoader(DEFAULT_TEMPLATE_PATH_LIST), extensions=['jinja2.ext.i18n'], **jenv_opt)
            server_env.install_gettext_translations(nbui, newstyle=False)
            identity_provider_kwargs = {'parent': self, 'log': self.log, 'login_handler_class': VoilaLoginHandler}
            if self.token is None and (not self.auto_token):
                identity_provider_kwargs['token'] = ''
            elif self.token is not None:
                identity_provider_kwargs['token'] = self.token
            self.identity_provider = self.identity_provider_class(**identity_provider_kwargs)
            self.authorizer = self.authorizer_class(parent=self, log=self.log, identity_provider=self.identity_provider)
        settings = dict(base_url=self.base_url, server_url=self.server_url or self.base_url, kernel_manager=self.kernel_manager, kernel_spec_manager=self.kernel_spec_manager, allow_remote_access=True, autoreload=self.autoreload, voila_jinja2_env=env, jinja2_env=server_env if JUPYTER_SERVER_2 else env, server_root_dir='/', contents_manager=self.contents_manager, config_manager=self.config_manager)
        if JUPYTER_SERVER_2:
            settings = {**settings, 'cookie_secret': self.cookie_secret, 'authorizer': self.authorizer, 'identity_provider': self.identity_provider, 'kernel_websocket_connection_class': self.kernel_websocket_connection_class, 'login_url': url_path_join(self.base_url, '/login')}
        return settings

    def init_handlers(self) -> List:
        """Initialize handlers for Voila application."""
        handlers = []
        handlers.extend([(url_path_join(self.server_url, '/api/kernels/%s' % _kernel_id_regex), KernelHandler), (url_path_join(self.server_url, '/api/kernels/%s/channels' % _kernel_id_regex), KernelWebsocketHandler if JUPYTER_SERVER_2 else ZMQChannelsHandler), (url_path_join(self.server_url, '/voila/templates/(.*)'), TemplateStaticFileHandler), (url_path_join(self.server_url, '/voila/static/(.*)'), MultiStaticFileHandler, {'paths': self.static_paths, 'default_filename': 'index.html'}), (url_path_join(self.server_url, '/voila/themes/(.*)'), ThemesHandler, {'themes_url': '/voila/themes', 'path': '', 'labextensions_path': jupyter_path('labextensions'), 'no_cache_paths': ['/']}), (url_path_join(self.server_url, '/voila/api/shutdown/(.*)'), VoilaShutdownKernelHandler)])
        if JUPYTER_SERVER_2:
            handlers.extend(self.identity_provider.get_handlers())
        if self.voila_configuration.preheat_kernel:
            handlers.append((url_path_join(self.server_url, '/voila/query/%s' % _kernel_id_regex), RequestInfoSocketHandler))
        if self.voila_configuration.enable_nbextensions:
            handlers.append((url_path_join(self.server_url, '/voila/nbextensions/(.*)'), FileFindHandler, {'path': self.nbextensions_path, 'no_cache_paths': ['/']}))
        handlers.append((url_path_join(self.server_url, '/voila/files/(.*)'), WhiteListFileHandler, {'whitelist': self.voila_configuration.file_whitelist, 'blacklist': self.voila_configuration.file_blacklist, 'path': self.root_dir}))
        tree_handler_conf = {'voila_configuration': self.voila_configuration}
        if self.notebook_path:
            handlers.append((url_path_join(self.server_url, '/(.*)'), VoilaHandler, {'notebook_path': os.path.relpath(self.notebook_path, self.root_dir), 'template_paths': self.template_paths, 'config': self.config, 'voila_configuration': self.voila_configuration, 'prelaunch_hook': self.prelaunch_hook}))
        else:
            self.log.debug('serving directory: %r', self.root_dir)
            handlers.extend([(self.server_url, VoilaTreeHandler, tree_handler_conf), (url_path_join(self.server_url, '/voila/tree' + path_regex), VoilaTreeHandler, tree_handler_conf), (url_path_join(self.server_url, '/voila/render/(.*)'), VoilaHandler, {'template_paths': self.template_paths, 'config': self.config, 'voila_configuration': self.voila_configuration, 'prelaunch_hook': self.prelaunch_hook})])
        return handlers

    def start(self):
        self.connection_dir = tempfile.mkdtemp(prefix='voila_', dir=self.connection_dir_root)
        self.log.info('Storing connection files in %s.' % self.connection_dir)
        self.log.info('Serving static files from %s.' % self.static_root)
        settings = self.init_settings()
        self.app = tornado.web.Application(**settings)
        self.app.settings.update(self.tornado_settings)
        handlers = self.init_handlers()
        self.app.add_handlers('.*$', handlers)
        self.listen()

    def _handle_signal_stop(self, sig, frame):
        self.log.info('Handle signal %s.' % sig)
        self.ioloop.add_callback_from_signal(self.ioloop.stop)

    def stop(self):
        shutil.rmtree(self.connection_dir)
        run_sync(self.kernel_manager.shutdown_all)()

    def random_ports(self, port, n):
        """Generate a list of n random ports near the given port.

        The first 5 ports will be sequential, and the remaining n-5 will be
        randomly selected in the range [port-2*n, port+2*n].
        """
        for i in range(min(5, n)):
            yield (port + i)
        for i in range(n - 5):
            yield max(1, port + random.randint(-2 * n, 2 * n))

    def listen(self):
        success = False
        for port in self.random_ports(self.port, self.port_retries + 1):
            try:
                self.app.listen(port, self.ip)
                self.port = port
                self.log.info('Voilà is running at:\n%s' % self.display_url)
            except socket.error as e:
                if e.errno == errno.EADDRINUSE:
                    self.log.info(_('The port %i is already in use, trying another port.') % port)
                    continue
                elif e.errno in (errno.EACCES, getattr(errno, 'WSAEACCES', errno.EACCES)):
                    self.log.warning(_('Permission to listen on port %i denied') % port)
                    continue
                else:
                    raise
            else:
                self.port = port
                success = True
                break
        if not success:
            self.log.critical(_('ERROR: the Voilà server could not be started because no available port could be found.'))
            self.exit(1)
        if self.open_browser:
            self.launch_browser()
        self.ioloop = tornado.ioloop.IOLoop.current()
        try:
            self.ioloop.start()
        except KeyboardInterrupt:
            self.log.info('Stopping...')
        finally:
            self.stop()

    def launch_browser(self):
        try:
            browser = webbrowser.get(self.browser or None)
        except webbrowser.Error as e:
            self.log.warning(_('No web browser found: %s.') % e)
            browser = None
        if not browser:
            return
        uri = self.base_url
        fd, open_file = tempfile.mkstemp(suffix='.html')
        with io.open(fd, 'w', encoding='utf-8') as fh:
            url = url_path_join(self.connection_url, uri)
            include_assets_functions = create_include_assets_functions(self.voila_configuration.template, url)
            jinja2_env = self.app.settings['voila_jinja2_env']
            template = jinja2_env.get_template('browser-open.html')
            fh.write(template.render(open_url=url, base_url=url, theme=self.voila_configuration.theme, **include_assets_functions))

        def target():
            return browser.open(urljoin('file:', pathname2url(open_file)), new=self.webbrowser_open_new)
        threading.Thread(target=target).start()