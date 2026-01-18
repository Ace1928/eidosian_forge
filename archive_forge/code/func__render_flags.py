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