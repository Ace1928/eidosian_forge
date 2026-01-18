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
def _gzip(bytestring):
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode='wb', compresslevel=3, mtime=0) as f:
        f.write(bytestring)
    return out.getvalue()