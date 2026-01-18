from __future__ import absolute_import
import datetime
import errno
from hashlib import sha1
import json
import logging
import os
import socket
import tempfile
import threading
import boto
import httplib2
import oauth2client.client
import oauth2client.service_account
from google_reauth import reauth_creds
import retry_decorator.retry_decorator
import six
from six import BytesIO
from six.moves import urllib
def _IsGCE():
    """Returns True if running on a GCE instance, otherwise False."""
    try:
        http = httplib2.Http()
        response, _ = http.request(METADATA_SERVER)
        return response.status == 200
    except (httplib2.ServerNotFoundError, socket.error):
        return False
    except Exception as e:
        LOG.warning("Failed to determine whether we're running on GCE, so we'llassume that we aren't: %s", e)
        return False
    return False