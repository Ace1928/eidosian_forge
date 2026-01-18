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
def _GetUserinfoUrl(credentials):
    url_root = 'https://oauth2.googleapis.com/tokeninfo'
    query_args = {'access_token': credentials.access_token}
    return '?'.join((url_root, urllib.parse.urlencode(query_args)))