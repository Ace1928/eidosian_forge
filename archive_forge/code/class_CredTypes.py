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
class CredTypes(object):
    HMAC = 'HMAC'
    OAUTH2_SERVICE_ACCOUNT = 'OAuth 2.0 Service Account'
    OAUTH2_USER_ACCOUNT = 'Oauth 2.0 User Account'
    GCE = 'GCE'