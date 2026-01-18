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
class GceAssertionCredentials(gce.AppAssertionCredentials):
    """Assertion credentials for GCE instances."""

    def __init__(self, scopes=None, service_account_name='default', **kwds):
        """Initializes the credentials instance.

        Args:
          scopes: The scopes to get. If None, whatever scopes that are
              available to the instance are used.
          service_account_name: The service account to retrieve the scopes
              from.
          **kwds: Additional keyword args.

        """
        self.__service_account_name = six.ensure_text(service_account_name, encoding='utf-8')
        cached_scopes = None
        cache_filename = kwds.get('cache_filename')
        if cache_filename:
            cached_scopes = self._CheckCacheFileForMatch(cache_filename, scopes)
        scopes = cached_scopes or self._ScopesFromMetadataServer(scopes)
        if cache_filename and (not cached_scopes):
            self._WriteCacheFile(cache_filename, scopes)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super(GceAssertionCredentials, self).__init__(scope=scopes, **kwds)

    @classmethod
    def Get(cls, *args, **kwds):
        try:
            return cls(*args, **kwds)
        except exceptions.Error:
            return None

    def _CheckCacheFileForMatch(self, cache_filename, scopes):
        """Checks the cache file to see if it matches the given credentials.

        Args:
          cache_filename: Cache filename to check.
          scopes: Scopes for the desired credentials.

        Returns:
          List of scopes (if cache matches) or None.
        """
        creds = {'scopes': sorted(list(scopes)) if scopes else None, 'svc_acct_name': self.__service_account_name}
        cache_file = _MultiProcessCacheFile(cache_filename)
        try:
            cached_creds_str = cache_file.LockedRead()
            if not cached_creds_str:
                return None
            cached_creds = json.loads(cached_creds_str)
            if creds['svc_acct_name'] == cached_creds['svc_acct_name']:
                if creds['scopes'] in (None, cached_creds['scopes']):
                    return cached_creds['scopes']
        except KeyboardInterrupt:
            raise
        except:
            pass

    def _WriteCacheFile(self, cache_filename, scopes):
        """Writes the credential metadata to the cache file.

        This does not save the credentials themselves (CredentialStore class
        optionally handles that after this class is initialized).

        Args:
          cache_filename: Cache filename to check.
          scopes: Scopes for the desired credentials.
        """
        scopes = sorted([six.ensure_text(scope) for scope in scopes])
        creds = {'scopes': scopes, 'svc_acct_name': self.__service_account_name}
        creds_str = json.dumps(creds)
        cache_file = _MultiProcessCacheFile(cache_filename)
        try:
            cache_file.LockedWrite(creds_str)
        except KeyboardInterrupt:
            raise
        except:
            pass

    def _ScopesFromMetadataServer(self, scopes):
        """Returns instance scopes based on GCE metadata server."""
        if not util.DetectGce():
            raise exceptions.ResourceUnavailableError('GCE credentials requested outside a GCE instance')
        if not self.GetServiceAccount(self.__service_account_name):
            raise exceptions.ResourceUnavailableError('GCE credentials requested but service account %s does not exist.' % self.__service_account_name)
        if scopes:
            scope_ls = util.NormalizeScopes(scopes)
            instance_scopes = self.GetInstanceScopes()
            if scope_ls > instance_scopes:
                raise exceptions.CredentialsError('Instance did not have access to scopes %s' % (sorted(list(scope_ls - instance_scopes)),))
        else:
            scopes = self.GetInstanceScopes()
        return scopes

    def GetServiceAccount(self, account):
        relative_url = 'instance/service-accounts'
        response = _GceMetadataRequest(relative_url)
        response_lines = [six.ensure_str(line).rstrip(u'/\n\r') for line in response.readlines()]
        return account in response_lines

    def GetInstanceScopes(self):
        relative_url = 'instance/service-accounts/{0}/scopes'.format(self.__service_account_name)
        response = _GceMetadataRequest(relative_url)
        return util.NormalizeScopes((six.ensure_str(scope).strip() for scope in response.readlines()))

    def _refresh(self, do_request):
        """Refresh self.access_token.

        This function replaces AppAssertionCredentials._refresh, which
        does not use the credential store and is therefore poorly
        suited for multi-threaded scenarios.

        Args:
          do_request: A function matching httplib2.Http.request's signature.

        """
        oauth2client.client.OAuth2Credentials._refresh(self, do_request)

    def _do_refresh_request(self, unused_http_request):
        """Refresh self.access_token by querying the metadata server.

        If self.store is initialized, store acquired credentials there.
        """
        relative_url = 'instance/service-accounts/{0}/token'.format(self.__service_account_name)
        try:
            response = _GceMetadataRequest(relative_url)
        except exceptions.CommunicationError:
            self.invalid = True
            if self.store:
                self.store.locked_put(self)
            raise
        content = six.ensure_str(response.read())
        try:
            credential_info = json.loads(content)
        except ValueError:
            raise exceptions.CredentialsError('Could not parse response as JSON: %s' % content)
        self.access_token = credential_info['access_token']
        if 'expires_in' in credential_info:
            expires_in = int(credential_info['expires_in'])
            self.token_expiry = datetime.timedelta(seconds=expires_in) + datetime.datetime.utcnow()
        else:
            self.token_expiry = None
        self.invalid = False
        if self.store:
            self.store.locked_put(self)

    def to_json(self):
        return super(gce.AppAssertionCredentials, self).to_json()

    @classmethod
    def from_json(cls, json_data):
        data = json.loads(json_data)
        kwargs = {}
        if 'cache_filename' in data.get('kwargs', []):
            kwargs['cache_filename'] = data['kwargs']['cache_filename']
        scope_list = None
        if 'scope' in data:
            scope_list = [data['scope']]
        credentials = GceAssertionCredentials(scopes=scope_list, **kwargs)
        if 'access_token' in data:
            credentials.access_token = data['access_token']
        if 'token_expiry' in data:
            credentials.token_expiry = datetime.datetime.strptime(data['token_expiry'], oauth2client.client.EXPIRY_FORMAT)
        if 'invalid' in data:
            credentials.invalid = data['invalid']
        return credentials

    @property
    def serialization_data(self):
        raise NotImplementedError('Cannot serialize credentials for GCE service accounts.')