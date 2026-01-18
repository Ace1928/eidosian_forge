import asyncio
import io
import inspect
import logging
import os
import queue
import uuid
import sys
import threading
import time
from typing_extensions import Literal
from werkzeug.serving import make_server
class JupyterDash:
    """
    Interact with dash apps inside jupyter notebooks.
    """
    default_mode: JupyterDisplayMode = 'inline'
    alive_token = str(uuid.uuid4())
    inline_exceptions: bool = True
    _servers = {}

    def infer_jupyter_proxy_config(self):
        """
        Infer the current Jupyter server configuration. This will detect
        the proper request_pathname_prefix and server_url values to use when
        displaying Dash apps.Dash requests will be routed through the proxy.

        Requirements:

        In the classic notebook, this method requires the `dash` nbextension
        which should be installed automatically with the installation of the
        jupyter-dash Python package. You can see what notebook extensions are installed
        by running the following command:
            $ jupyter nbextension list

        In JupyterLab, this method requires the `@plotly/dash-jupyterlab` labextension. This
        extension should be installed automatically with the installation of the
        jupyter-dash Python package, but JupyterLab must be allowed to rebuild before
        the extension is activated (JupyterLab should automatically detect the
        extension and produce a popup dialog asking for permission to rebuild). You can
        see what JupyterLab extensions are installed by running the following command:
            $ jupyter labextension list
        """
        if not self.in_ipython or self.in_colab:
            return
        _request_jupyter_config()

    def __init__(self):
        self.in_ipython = get_ipython() is not None
        self.in_colab = 'google.colab' in sys.modules
        if _dep_installed and self.in_ipython and _dash_comm:

            @_dash_comm.on_msg
            def _receive_message(msg):
                prev_parent = _caller.get('parent')
                if prev_parent and prev_parent != _dash_comm.kernel.get_parent():
                    _dash_comm.kernel.set_parent([prev_parent['header']['session']], prev_parent)
                    del _caller['parent']
                msg_data = msg.get('content').get('data')
                msg_type = msg_data.get('type', None)
                if msg_type == 'base_url_response':
                    _jupyter_config.update(msg_data)

    def run_app(self, app, mode: JupyterDisplayMode=None, width='100%', height=650, host='127.0.0.1', port=8050, server_url=None):
        """
        :type app: dash.Dash
        :param mode: How to display the app on the notebook. One Of:
            ``"external"``: The URL of the app will be displayed in the notebook
                output cell. Clicking this URL will open the app in the default
                web browser.
            ``"inline"``: The app will be displayed inline in the notebook output cell
                in an iframe.
            ``"jupyterlab"``: The app will be displayed in a dedicate tab in the
                JupyterLab interface. Requires JupyterLab and the `jupyterlab-dash`
                extension.
        :param width: Width of app when displayed using mode="inline"
        :param height: Height of app when displayed using mode="inline"
        :param host: Host of the server
        :param port: Port used by the server
        :param server_url: Use if a custom url is required to display the app.
        """
        if self.in_colab:
            valid_display_values = ['inline', 'external']
        else:
            valid_display_values = ['jupyterlab', 'inline', 'external', 'tab', '_none']
        if mode is None:
            mode = self.default_mode
        elif not isinstance(mode, str):
            raise ValueError(f'The mode argument must be a string\n    Received value of type {type(mode)}: {repr(mode)}')
        else:
            mode = mode.lower()
            if mode not in valid_display_values:
                raise ValueError(f'Invalid display argument {mode}\n    Valid arguments: {valid_display_values}')
        old_server = self._servers.get((host, port))
        if old_server:
            old_server.shutdown()
            del self._servers[host, port]
        if 'base_subpath' in _jupyter_config:
            requests_pathname_prefix = _jupyter_config['base_subpath'].rstrip('/') + '/proxy/{port}/'
        else:
            requests_pathname_prefix = app.config.get('requests_pathname_prefix', None)
        if requests_pathname_prefix is not None:
            requests_pathname_prefix = requests_pathname_prefix.format(port=port)
        else:
            requests_pathname_prefix = '/'
        dict.__setitem__(app.config, 'requests_pathname_prefix', requests_pathname_prefix)
        if server_url is None:
            if 'server_url' in _jupyter_config:
                server_url = _jupyter_config['server_url'].rstrip('/')
            else:
                domain_base = os.environ.get('DASH_DOMAIN_BASE', None)
                if domain_base:
                    server_url = 'https://' + domain_base
                else:
                    server_url = f'http://{host}:{port}'
        else:
            server_url = server_url.rstrip('/')
        dashboard_url = f'{server_url}{requests_pathname_prefix}'
        try:
            import orjson
        except ImportError:
            pass
        err_q = queue.Queue()
        server = make_server(host, port, app.server, threaded=True, processes=0)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        @retry(stop_max_attempt_number=15, wait_exponential_multiplier=100, wait_exponential_max=1000)
        def run():
            try:
                server.serve_forever()
            except SystemExit:
                pass
            except Exception as error:
                err_q.put(error)
                raise error
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
        self._servers[host, port] = server
        alive_url = f'http://{host}:{port}/_alive_{JupyterDash.alive_token}'

        def _get_error():
            try:
                err = err_q.get_nowait()
                if err:
                    raise err
            except queue.Empty:
                pass

        @retry(stop_max_attempt_number=15, wait_exponential_multiplier=10, wait_exponential_max=1000)
        def wait_for_app():
            _get_error()
            try:
                req = requests.get(alive_url)
                res = req.content.decode()
                if req.status_code != 200:
                    raise Exception(res)
                if res != 'Alive':
                    url = f'http://{host}:{port}'
                    raise OSError(f"Address '{url}' already in use.\n    Try passing a different port to run_server.")
            except requests.ConnectionError as err:
                _get_error()
                raise err
        try:
            wait_for_app()
            if self.in_colab:
                JupyterDash._display_in_colab(dashboard_url, port, mode, width, height)
            else:
                JupyterDash._display_in_jupyter(dashboard_url, port, mode, width, height)
        except Exception as final_error:
            msg = str(final_error)
            if msg.startswith('<!'):
                display(HTML(msg))
            else:
                raise final_error

    @staticmethod
    def _display_in_colab(dashboard_url, port, mode, width, height):
        from google.colab import output
        if mode == 'inline':
            output.serve_kernel_port_as_iframe(port, width=width, height=height)
        elif mode == 'external':
            print('Dash app running on:')
            output.serve_kernel_port_as_window(port, anchor_text=dashboard_url)

    @staticmethod
    def _display_in_jupyter(dashboard_url, port, mode, width, height):
        if mode == 'inline':
            display(IFrame(dashboard_url, width, height))
        elif mode in ('external', 'tab'):
            print(f'Dash app running on {dashboard_url}')
            if mode == 'tab':
                display(Javascript(f"window.open('{dashboard_url}')"))
        elif mode == 'jupyterlab':
            _dash_comm.send({'type': 'show', 'port': port, 'url': dashboard_url})

    @staticmethod
    def serve_alive():
        return 'Alive'

    def configure_callback_exception_handling(self, app, dev_tools_prune_errors):
        """Install traceback handling for callbacks"""

        @app.server.errorhandler(Exception)
        def _wrap_errors(error):
            skip = _get_skip(error) if dev_tools_prune_errors else 0
            original_formatargvalues = inspect.formatargvalues
            inspect.formatargvalues = _custom_formatargvalues
            try:
                ostream = io.StringIO()
                ipytb = FormattedTB(tb_offset=skip, mode='Verbose', color_scheme='NoColor', include_vars=True, ostream=ostream)
                ipytb()
            finally:
                inspect.formatargvalues = original_formatargvalues
            stacktrace = ostream.getvalue()
            if self.inline_exceptions:
                print(stacktrace)
            return (stacktrace, 500)

    @property
    def active(self):
        _inside_dbx = 'DATABRICKS_RUNTIME_VERSION' in os.environ
        return _dep_installed and (not _inside_dbx) and (self.in_ipython or self.in_colab)