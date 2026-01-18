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
def _serve_notifications(self, request):
    """Serve JSON payload of notifications to show in the UI."""
    response = utils.redirect('../notifications_note.json')
    response.autocorrect_location_header = False
    return response