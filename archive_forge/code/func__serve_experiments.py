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
def _serve_experiments(self, request):
    """Serve a JSON array of experiments.

        Experiments are ordered by experiment started time (aka first
        event time) with empty times sorted last, and then ties are
        broken by sorting on the experiment name.
        """
    results = self.list_experiments_impl()
    return http_util.Respond(request, results, 'application/json')