import argparse
import functools
import gzip
import io
import mimetypes
import posixpath
import zipfile
from werkzeug import utils
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard import version
class CorePlugin(base_plugin.TBPlugin):
    """Core plugin for TensorBoard.

    This plugin serves runs, configuration data, and static assets. This
    plugin should always be present in a TensorBoard WSGI application.
    """
    plugin_name = 'core'

    def __init__(self, context, include_debug_info=None):
        """Instantiates CorePlugin.

        Args:
          context: A base_plugin.TBContext instance.
          include_debug_info: If true, `/data/environment` will include some
            basic information like the TensorBoard server version. Disabled by
            default to prevent surprising information leaks in custom builds of
            TensorBoard.
        """
        self._flags = context.flags
        logdir_spec = context.flags.logdir_spec if context.flags else ''
        self._logdir = context.logdir or logdir_spec
        self._window_title = context.window_title
        self._path_prefix = context.flags.path_prefix if context.flags else None
        self._assets_zip_provider = context.assets_zip_provider
        self._data_provider = context.data_provider
        self._include_debug_info = bool(include_debug_info)

    def is_active(self):
        return True

    def get_plugin_apps(self):
        apps = {'/___rPc_sWiTcH___': self._send_404_without_logging, '/audio': self._redirect_to_index, '/data/environment': self._serve_environment, '/data/logdir': self._serve_logdir, '/data/runs': self._serve_runs, '/data/experiments': self._serve_experiments, '/data/experiment_runs': self._serve_experiment_runs, '/data/notifications': self._serve_notifications, '/data/window_properties': self._serve_window_properties, '/events': self._redirect_to_index, '/favicon.ico': self._send_404_without_logging, '/graphs': self._redirect_to_index, '/histograms': self._redirect_to_index, '/images': self._redirect_to_index}
        apps.update(self.get_resource_apps())
        return apps

    def get_resource_apps(self):
        apps = {}
        if not self._assets_zip_provider:
            return apps
        with self._assets_zip_provider() as fp:
            with zipfile.ZipFile(fp) as zip_:
                for path in zip_.namelist():
                    content = zip_.read(path)
                    if path == 'index.html':
                        apps['/' + path] = functools.partial(self._serve_index, content)
                        continue
                    gzipped_asset_bytes = _gzip(content)
                    wsgi_app = functools.partial(self._serve_asset, path, gzipped_asset_bytes)
                    apps['/' + path] = wsgi_app
        apps['/'] = apps['/index.html']
        return apps

    @wrappers.Request.application
    def _send_404_without_logging(self, request):
        return http_util.Respond(request, 'Not found', 'text/plain', code=404)

    @wrappers.Request.application
    def _redirect_to_index(self, unused_request):
        return utils.redirect('/')

    @wrappers.Request.application
    def _serve_asset(self, path, gzipped_asset_bytes, request):
        """Serves a pre-gzipped static asset from the zip file."""
        mimetype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        expires = JS_CACHE_EXPIRATION_IN_SECS if request.args.get('_file_hash') and mimetype in JS_MIMETYPES else 0
        return http_util.Respond(request, gzipped_asset_bytes, mimetype, content_encoding='gzip', expires=expires)

    @wrappers.Request.application
    def _serve_index(self, index_asset_bytes, request):
        """Serves index.html content.

        Note that we opt out of gzipping index.html to write preamble before the
        resource content. This inflates the resource size from 2x kiB to 1xx
        kiB, but we require an ability to flush preamble with the HTML content.
        """
        relpath = posixpath.relpath(self._path_prefix, request.script_root) if self._path_prefix else '.'
        meta_header = '<!doctype html><meta name="tb-relative-root" content="%s/">' % relpath
        content = meta_header.encode('utf-8') + index_asset_bytes
        return http_util.Respond(request, content, 'text/html', content_encoding='identity')

    @wrappers.Request.application
    def _serve_environment(self, request):
        """Serve a JSON object describing the TensorBoard parameters."""
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        md = self._data_provider.experiment_metadata(ctx, experiment_id=experiment)
        environment = {'version': version.VERSION, 'data_location': md.data_location, 'window_title': self._window_title, 'experiment_name': md.experiment_name, 'experiment_description': md.experiment_description, 'creation_time': md.creation_time}
        if self._include_debug_info:
            environment['debug'] = {'data_provider': str(self._data_provider), 'flags': self._render_flags()}
        return http_util.Respond(request, environment, 'application/json')

    def _render_flags(self):
        """Return a JSON-and-human-friendly version of `self._flags`.

        Like `json.loads(json.dumps(self._flags, default=str))` but
        without the wasteful serialization overhead.
        """
        if self._flags is None:
            return None

        def go(x):
            if isinstance(x, (type(None), str, int, float)):
                return x
            if isinstance(x, (list, tuple)):
                return [go(v) for v in x]
            if isinstance(x, dict):
                return {str(k): go(v) for k, v in x.items()}
            return str(x)
        return go(vars(self._flags))

    @wrappers.Request.application
    def _serve_logdir(self, request):
        """Respond with a JSON object containing this TensorBoard's logdir."""
        return http_util.Respond(request, {'logdir': self._logdir}, 'application/json')

    @wrappers.Request.application
    def _serve_window_properties(self, request):
        """Serve a JSON object containing this TensorBoard's window
        properties."""
        return http_util.Respond(request, {'window_title': self._window_title}, 'application/json')

    @wrappers.Request.application
    def _serve_runs(self, request):
        """Serve a JSON array of run names, ordered by run started time.

        Sort order is by started time (aka first event time) with empty
        times sorted last, and then ties are broken by sorting on the
        run name.
        """
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        runs = sorted(self._data_provider.list_runs(ctx, experiment_id=experiment), key=lambda run: (run.start_time if run.start_time is not None else float('inf'), run.run_name))
        run_names = [run.run_name for run in runs]
        return http_util.Respond(request, run_names, 'application/json')

    @wrappers.Request.application
    def _serve_experiments(self, request):
        """Serve a JSON array of experiments.

        Experiments are ordered by experiment started time (aka first
        event time) with empty times sorted last, and then ties are
        broken by sorting on the experiment name.
        """
        results = self.list_experiments_impl()
        return http_util.Respond(request, results, 'application/json')

    def list_experiments_impl(self):
        return []

    @wrappers.Request.application
    def _serve_experiment_runs(self, request):
        """Serve a JSON runs of an experiment, specified with query param
        `experiment`, with their nested data, tag, populated.

        Runs returned are ordered by started time (aka first event time)
        with empty times sorted last, and then ties are broken by
        sorting on the run name. Tags are sorted by its name,
        displayName, and lastly, inserted time.
        """
        results = []
        return http_util.Respond(request, results, 'application/json')

    @wrappers.Request.application
    def _serve_notifications(self, request):
        """Serve JSON payload of notifications to show in the UI."""
        response = utils.redirect('../notifications_note.json')
        response.autocorrect_location_header = False
        return response