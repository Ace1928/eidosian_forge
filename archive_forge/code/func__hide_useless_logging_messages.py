import logging
import os
import sys
import warnings
from cliff import app
from cliff import commandmanager
from keystoneauth1 import loading
from oslo_utils import importutils
from vitrageclient import __version__
from vitrageclient import auth
from vitrageclient import client
from vitrageclient.v1.cli import alarm
from vitrageclient.v1.cli import event
from vitrageclient.v1.cli import healthcheck
from vitrageclient.v1.cli import rca
from vitrageclient.v1.cli import resource
from vitrageclient.v1.cli import service
from vitrageclient.v1.cli import status
from vitrageclient.v1.cli import template
from vitrageclient.v1.cli import topology
from vitrageclient.v1.cli import webhook
def _hide_useless_logging_messages(self):
    requests_log = logging.getLogger('requests')
    cliff_log = logging.getLogger('cliff')
    stevedore_log = logging.getLogger('stevedore')
    iso8601_log = logging.getLogger('iso8601')
    cliff_log.setLevel(logging.ERROR)
    stevedore_log.setLevel(logging.ERROR)
    iso8601_log.setLevel(logging.ERROR)
    if self.options.debug:
        requests_log.setLevel(logging.DEBUG)
    else:
        requests_log.setLevel(logging.ERROR)