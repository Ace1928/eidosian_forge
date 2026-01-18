import errno
import logging
import os
import subprocess
import tempfile
import time
import grpc
import pkg_resources
from tensorboard.data import grpc_provider
from tensorboard.data import ingester
from tensorboard.data.proto import data_provider_pb2
from tensorboard.util import tb_logging
def _maybe_read_file(path):
    """Read a file, or return `None` on ENOENT specifically."""
    try:
        with open(path) as infile:
            return infile.read()
    except OSError as e:
        if e.errno == errno.ENOENT:
            return None
        raise