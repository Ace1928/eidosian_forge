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
def _nonnegative_float(v):
    try:
        v = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError('invalid float: %r' % v)
    if not v >= 0:
        raise argparse.ArgumentTypeError('must be non-negative: %r' % v)
    return v