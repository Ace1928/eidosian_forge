from __future__ import print_function
import argparse
import contextlib
import datetime
import json
import os
import threading
import warnings
import httplib2
import oauth2client
import oauth2client.client
from oauth2client import service_account
from oauth2client import tools  # for gflags declarations
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.py import exceptions
from apitools.base.py import util
def _GetRunFlowFlags(args=None):
    """Retrieves command line flags based on gflags module."""
    parser = argparse.ArgumentParser(parents=[tools.argparser])
    flags, _ = parser.parse_known_args(args=args)
    if hasattr(FLAGS, 'auth_host_name'):
        flags.auth_host_name = FLAGS.auth_host_name
    if hasattr(FLAGS, 'auth_host_port'):
        flags.auth_host_port = FLAGS.auth_host_port
    if hasattr(FLAGS, 'auth_local_webserver'):
        flags.noauth_local_webserver = not FLAGS.auth_local_webserver
    return flags