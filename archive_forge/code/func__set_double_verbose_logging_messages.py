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
@staticmethod
def _set_double_verbose_logging_messages(root_logger):
    root_logger.setLevel(logging.DEBUG)