import base64
import collections
import hashlib
import io
import json
import re
import textwrap
import time
from urllib import parse as urlparse
import zipfile
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import auth_context_middleware
from tensorboard.backend import client_feature_flags
from tensorboard.backend import empty_path_redirect
from tensorboard.backend import experiment_id
from tensorboard.backend import experimental_plugin
from tensorboard.backend import http_util
from tensorboard.backend import path_prefix
from tensorboard.backend import security_validator
from tensorboard.plugins import base_plugin
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
@wrappers.Request.application
def _serve_plugins_listing(self, request):
    """Serves an object mapping plugin name to whether it is enabled.

        Args:
          request: The werkzeug.Request object.

        Returns:
          A werkzeug.Response object.
        """
    response = collections.OrderedDict()
    ctx = plugin_util.context(request.environ)
    eid = plugin_util.experiment_id(request.environ)
    plugins_with_data = frozenset(self._data_provider.list_plugins(ctx, experiment_id=eid) or frozenset() if self._data_provider is not None else frozenset())
    plugins_to_skip = self._experimental_plugins - frozenset(request.args.getlist(EXPERIMENTAL_PLUGINS_QUERY_PARAM))
    for plugin in self._plugins:
        if plugin.plugin_name in plugins_to_skip:
            continue
        if type(plugin) is core_plugin.CorePlugin:
            continue
        is_active = bool(frozenset(plugin.data_plugin_names()) & plugins_with_data)
        if not is_active:
            try:
                start = time.time()
                is_active = plugin.is_active()
                elapsed = time.time() - start
                logger.info('Plugin listing: is_active() for %s took %0.3f seconds', plugin.plugin_name, elapsed)
            except Exception:
                is_active = False
                logger.error('Plugin listing: is_active() for %s failed (marking inactive)', plugin.plugin_name, exc_info=True)
        plugin_metadata = plugin.frontend_metadata()
        output_metadata = {'disable_reload': plugin_metadata.disable_reload, 'enabled': is_active, 'remove_dom': plugin_metadata.remove_dom}
        if plugin_metadata.tab_name is not None:
            output_metadata['tab_name'] = plugin_metadata.tab_name
        else:
            output_metadata['tab_name'] = plugin.plugin_name
        es_module_handler = plugin_metadata.es_module_path
        element_name = plugin_metadata.element_name
        is_ng_component = plugin_metadata.is_ng_component
        if is_ng_component:
            if element_name is not None:
                raise ValueError('Plugin %r declared as both Angular built-in and legacy' % plugin.plugin_name)
            if es_module_handler is not None:
                raise ValueError('Plugin %r declared as both Angular built-in and iframed' % plugin.plugin_name)
            loading_mechanism = {'type': 'NG_COMPONENT'}
        elif element_name is not None and es_module_handler is not None:
            logger.error('Plugin %r declared as both legacy and iframed; skipping', plugin.plugin_name)
            continue
        elif element_name is not None and es_module_handler is None:
            loading_mechanism = {'type': 'CUSTOM_ELEMENT', 'element_name': element_name}
        elif element_name is None and es_module_handler is not None:
            loading_mechanism = {'type': 'IFRAME', 'module_path': ''.join([request.script_root, DATA_PREFIX, PLUGIN_PREFIX, '/', plugin.plugin_name, es_module_handler])}
        else:
            loading_mechanism = {'type': 'NONE'}
        output_metadata['loading_mechanism'] = loading_mechanism
        response[plugin.plugin_name] = output_metadata
    return http_util.Respond(request, response, 'application/json')