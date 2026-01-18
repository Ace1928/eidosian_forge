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
class OAuth2GCEClient(OAuth2Client):
    """OAuth2 client for GCE instance."""

    def __init__(self):
        super(OAuth2GCEClient, self).__init__(cache_key_base='', access_token_cache=InMemoryTokenCache())

    @retry_decorator.retry(GsAccessTokenRefreshError, tries=6, timeout_secs=1)
    def FetchAccessToken(self, rapt_token=None):
        """Fetches an access token from the provider's token endpoint.

    Fetches an access token from the GCE metadata server.

    Args:
      rapt_token: (str) Ignored for this class. Service accounts don't use
          reauth credentials.

    Returns:
      The fetched AccessToken.
    """
        del rapt_token
        response = None
        try:
            http = httplib2.Http()
            response, content = http.request(META_TOKEN_URI, method='GET', body=None, headers=META_HEADERS)
            content = six.ensure_text(content)
        except Exception as e:
            raise GsAccessTokenRefreshError(e)
        if response.status == 200:
            d = json.loads(content)
            return AccessToken(d['access_token'], datetime.datetime.now() + datetime.timedelta(seconds=d.get('expires_in', 0)), datetime_strategy=self.datetime_strategy, rapt_token=None)