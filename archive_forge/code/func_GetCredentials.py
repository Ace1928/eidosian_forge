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
def GetCredentials(self):
    """Fetches a credentials objects from the provider's token endpoint."""
    access_token = self.GetAccessToken()
    credentials = reauth_creds.Oauth2WithReauthCredentials(access_token.token, self.client_id, self.client_secret, self.refresh_token, access_token.expiry, self.token_uri, None)
    return credentials