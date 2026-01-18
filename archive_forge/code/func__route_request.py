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
def _route_request(self, environ, start_response):
    """Delegate an incoming request to sub-applications.

        This method supports strict string matching and wildcard routes of a
        single path component, such as `/foo/*`. Other routing patterns,
        like regular expressions, are not supported.

        This is the main TensorBoard entry point before middleware is
        applied. (See `_create_wsgi_app`.)

        Args:
          environ: See WSGI spec (PEP 3333).
          start_response: See WSGI spec (PEP 3333).
        """
    request = wrappers.Request(environ)
    parsed_url = urlparse.urlparse(request.path)
    clean_path = _clean_path(parsed_url.path)
    if clean_path in self.exact_routes:
        return self.exact_routes[clean_path](environ, start_response)
    else:
        for path_prefix in self.prefix_routes:
            if clean_path.startswith(path_prefix):
                return self.prefix_routes[path_prefix](environ, start_response)
        logger.warning('path %s not found, sending 404', clean_path)
        return http_util.Respond(request, 'Not found', 'text/plain', code=404)(environ, start_response)