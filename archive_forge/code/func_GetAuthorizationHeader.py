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
def GetAuthorizationHeader(self):
    """Gets the access token HTTP authorization header value.

    Returns:
      The value of an Authorization HTTP header that authenticates
      requests with an OAuth2 access token.
    """
    return 'Bearer %s' % self.GetAccessToken().token