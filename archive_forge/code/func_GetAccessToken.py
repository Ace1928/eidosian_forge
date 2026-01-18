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
def GetAccessToken(self):
    """Obtains an access token for this client.

    This client's access token cache is first checked for an existing,
    not-yet-expired access token. If none is found, the client obtains a fresh
    access token from the OAuth2 provider's token endpoint.

    Returns:
      The cached or freshly obtained AccessToken.
    Raises:
      oauth2client.client.AccessTokenRefreshError if an error occurs.
    """
    token_exchange_lock.acquire()
    try:
        cache_key = self.CacheKey()
        LOG.debug('GetAccessToken: checking cache for key %s', cache_key)
        access_token = self.access_token_cache.GetToken(cache_key)
        LOG.debug('GetAccessToken: token from cache: %s', access_token)
        if access_token is None or access_token.ShouldRefresh():
            rapt = None if access_token is None else access_token.rapt_token
            LOG.debug('GetAccessToken: fetching fresh access token...')
            access_token = self.FetchAccessToken(rapt_token=rapt)
            LOG.debug('GetAccessToken: fresh access token: %s', access_token)
            self.access_token_cache.PutToken(cache_key, access_token)
        return access_token
    finally:
        token_exchange_lock.release()