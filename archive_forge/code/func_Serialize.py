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
def Serialize(self):
    """Serializes this object as URI-encoded key-value pairs."""
    kv = {'token': self.token}
    if self.expiry:
        t = self.expiry
        tupl = (t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond)
        kv['expiry'] = ','.join([str(i) for i in tupl])
    if self.rapt_token:
        kv['rapt_token'] = self.rapt_token
    return urllib.parse.urlencode(kv)