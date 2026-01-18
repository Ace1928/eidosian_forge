from concurrent import futures
import enum
import errno
import io
import logging as pylogging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import eventlet
from eventlet import patcher
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import comm
def _fd_logger(level=logging.WARN):
    """Helper that returns a file object that is asynchronously logged"""
    read_fd, write_fd = os.pipe()
    read_end = fdopen(read_fd, 'r', 1)
    write_end = fdopen(write_fd, 'w', 1)

    def logger(f):
        for line in f:
            LOG.log(level, 'privsep log: %s', line.rstrip())
    t = threading.Thread(name='fd_logger', target=logger, args=(read_end,))
    t.daemon = True
    t.start()
    return write_end