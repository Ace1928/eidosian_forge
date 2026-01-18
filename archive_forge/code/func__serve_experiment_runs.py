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