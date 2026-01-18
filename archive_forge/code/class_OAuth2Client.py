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
class OAuth2Client(object):
    """Common logic for OAuth2 clients."""

    def __init__(self, cache_key_base, access_token_cache=None, datetime_strategy=datetime.datetime, auth_uri=None, token_uri=None, disable_ssl_certificate_validation=False, proxy_host=None, proxy_port=None, proxy_user=None, proxy_pass=None, ca_certs_file=None):
        self.auth_uri = auth_uri
        self.token_uri = token_uri
        self.cache_key_base = cache_key_base
        self.datetime_strategy = datetime_strategy
        self.access_token_cache = access_token_cache or InMemoryTokenCache()
        self.disable_ssl_certificate_validation = disable_ssl_certificate_validation
        self.ca_certs_file = ca_certs_file
        if proxy_host and proxy_port:
            self._proxy_info = httplib2.ProxyInfo(httplib2.socks.PROXY_TYPE_HTTP, proxy_host, proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass, proxy_rdns=True)
        else:
            self._proxy_info = None

    def CreateHttpRequest(self):
        return httplib2.Http(ca_certs=self.ca_certs_file, disable_ssl_certificate_validation=self.disable_ssl_certificate_validation, proxy_info=self._proxy_info)

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

    def CacheKey(self):
        """Computes a cache key.

    The cache key is computed as the SHA1 hash of the refresh token for user
    accounts, or the hash of the gs_service_client_id for service accounts,
    which satisfies the FileSystemTokenCache requirement that cache keys do not
    leak information about token values.

    Returns:
      A hash key.
    """
        h = sha1()
        if isinstance(self.cache_key_base, six.text_type):
            val = self.cache_key_base.encode('utf-8')
        else:
            val = self.cache_key_base
        h.update(val)
        return h.hexdigest()

    def GetAuthorizationHeader(self):
        """Gets the access token HTTP authorization header value.

    Returns:
      The value of an Authorization HTTP header that authenticates
      requests with an OAuth2 access token.
    """
        return 'Bearer %s' % self.GetAccessToken().token