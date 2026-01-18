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
def _create_wsgi_app(self):
    """Apply middleware to create the final WSGI app."""
    app = self._route_request
    for middleware in self._extra_middlewares:
        app = middleware(app)
    app = auth_context_middleware.AuthContextMiddleware(app, self._auth_providers)
    app = client_feature_flags.ClientFeatureFlagsMiddleware(app)
    app = empty_path_redirect.EmptyPathRedirectMiddleware(app)
    app = experiment_id.ExperimentIdMiddleware(app)
    app = path_prefix.PathPrefixMiddleware(app, self._path_prefix)
    app = security_validator.SecurityValidatorMiddleware(app)
    app = _handling_errors(app)
    return app