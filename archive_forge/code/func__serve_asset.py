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
@wrappers.Request.application
def _serve_asset(self, path, gzipped_asset_bytes, request):
    """Serves a pre-gzipped static asset from the zip file."""
    mimetype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    expires = JS_CACHE_EXPIRATION_IN_SECS if request.args.get('_file_hash') and mimetype in JS_MIMETYPES else 0
    return http_util.Respond(request, gzipped_asset_bytes, mimetype, content_encoding='gzip', expires=expires)