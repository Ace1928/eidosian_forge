import abc
import errno
import os
import signal
import sys
import time
import eventlet
from eventlet.green import socket
from eventlet.green import ssl
import eventlet.greenio
import eventlet.wsgi
import functools
from oslo_concurrency import processutils
from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from paste.deploy import loadwsgi
from routes import middleware
import webob.dec
import webob.exc
from heat.api.aws import exception as aws_exception
from heat.common import exception
from heat.common.i18n import _
from heat.common import serializers
def _pipe_watcher(self):

    def _on_timeout_exit(*args):
        LOG.info('Graceful shutdown timeout exceeded, instantaneous exiting')
        os._exit(1)
    self.readpipe.read(1)
    LOG.info('Parent process has died unexpectedly, exiting')
    signal.signal(signal.SIGALRM, _on_timeout_exit)
    signal.alarm(1)
    eventlet.wsgi.is_accepting = False
    self.sock.close()
    sys.exit(1)