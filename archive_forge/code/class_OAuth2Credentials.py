import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
class OAuth2Credentials(Credentials):
    """Credentials object for OAuth 2.0.

    Credentials can be applied to an httplib2.Http object using the authorize()
    method, which then adds the OAuth 2.0 access token to each request.

    OAuth2Credentials objects may be safely pickled and unpickled.
    """

    @_helpers.positional(8)
    def __init__(self, access_token, client_id, client_secret, refresh_token, token_expiry, token_uri, user_agent, revoke_uri=None, id_token=None, token_response=None, scopes=None, token_info_uri=None, id_token_jwt=None):
        """Create an instance of OAuth2Credentials.

        This constructor is not usually called by the user, instead
        OAuth2Credentials objects are instantiated by the OAuth2WebServerFlow.

        Args:
            access_token: string, access token.
            client_id: string, client identifier.
            client_secret: string, client secret.
            refresh_token: string, refresh token.
            token_expiry: datetime, when the access_token expires.
            token_uri: string, URI of token endpoint.
            user_agent: string, The HTTP User-Agent to provide for this
                        application.
            revoke_uri: string, URI for revoke endpoint. Defaults to None; a
                        token can't be revoked if this is None.
            id_token: object, The identity of the resource owner.
            token_response: dict, the decoded response to the token request.
                            None if a token hasn't been requested yet. Stored
                            because some providers (e.g. wordpress.com) include
                            extra fields that clients may want.
            scopes: list, authorized scopes for these credentials.
            token_info_uri: string, the URI for the token info endpoint.
                            Defaults to None; scopes can not be refreshed if
                            this is None.
            id_token_jwt: string, the encoded and signed identity JWT. The
                          decoded version of this is stored in id_token.

        Notes:
            store: callable, A callable that when passed a Credential
                   will store the credential back to where it came from.
                   This is needed to store the latest access_token if it
                   has expired and been refreshed.
        """
        self.access_token = access_token
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.store = None
        self.token_expiry = token_expiry
        self.token_uri = token_uri
        self.user_agent = user_agent
        self.revoke_uri = revoke_uri
        self.id_token = id_token
        self.id_token_jwt = id_token_jwt
        self.token_response = token_response
        self.scopes = set(_helpers.string_to_scopes(scopes or []))
        self.token_info_uri = token_info_uri
        self.invalid = False

    def authorize(self, http):
        """Authorize an httplib2.Http instance with these credentials.

        The modified http.request method will add authentication headers to
        each request and will refresh access_tokens when a 401 is received on a
        request. In addition the http.request method has a credentials
        property, http.request.credentials, which is the Credentials object
        that authorized it.

        Args:
            http: An instance of ``httplib2.Http`` or something that acts
                  like it.

        Returns:
            A modified instance of http that was passed in.

        Example::

            h = httplib2.Http()
            h = credentials.authorize(h)

        You can't create a new OAuth subclass of httplib2.Authentication
        because it never gets passed the absolute URI, which is needed for
        signing. So instead we have to overload 'request' with a closure
        that adds in the Authorization header and then calls the original
        version of 'request()'.
        """
        transport.wrap_http_for_auth(self, http)
        return http

    def refresh(self, http):
        """Forces a refresh of the access_token.

        Args:
            http: httplib2.Http, an http object to be used to make the refresh
                  request.
        """
        self._refresh(http)

    def revoke(self, http):
        """Revokes a refresh_token and makes the credentials void.

        Args:
            http: httplib2.Http, an http object to be used to make the revoke
                  request.
        """
        self._revoke(http)

    def apply(self, headers):
        """Add the authorization to the headers.

        Args:
            headers: dict, the headers to add the Authorization header to.
        """
        headers['Authorization'] = 'Bearer ' + self.access_token

    def has_scopes(self, scopes):
        """Verify that the credentials are authorized for the given scopes.

        Returns True if the credentials authorized scopes contain all of the
        scopes given.

        Args:
            scopes: list or string, the scopes to check.

        Notes:
            There are cases where the credentials are unaware of which scopes
            are authorized. Notably, credentials obtained and stored before
            this code was added will not have scopes, AccessTokenCredentials do
            not have scopes. In both cases, you can use refresh_scopes() to
            obtain the canonical set of scopes.
        """
        scopes = _helpers.string_to_scopes(scopes)
        return set(scopes).issubset(self.scopes)

    def retrieve_scopes(self, http):
        """Retrieves the canonical list of scopes for this access token.

        Gets the scopes from the OAuth2 provider.

        Args:
            http: httplib2.Http, an http object to be used to make the refresh
                  request.

        Returns:
            A set of strings containing the canonical list of scopes.
        """
        self._retrieve_scopes(http)
        return self.scopes

    @classmethod
    def from_json(cls, json_data):
        """Instantiate a Credentials object from a JSON description of it.

        The JSON should have been produced by calling .to_json() on the object.

        Args:
            json_data: string or bytes, JSON to deserialize.

        Returns:
            An instance of a Credentials subclass.
        """
        data = json.loads(_helpers._from_bytes(json_data))
        if data.get('token_expiry') and (not isinstance(data['token_expiry'], datetime.datetime)):
            try:
                data['token_expiry'] = datetime.datetime.strptime(data['token_expiry'], EXPIRY_FORMAT)
            except ValueError:
                data['token_expiry'] = None
        retval = cls(data['access_token'], data['client_id'], data['client_secret'], data['refresh_token'], data['token_expiry'], data['token_uri'], data['user_agent'], revoke_uri=data.get('revoke_uri', None), id_token=data.get('id_token', None), id_token_jwt=data.get('id_token_jwt', None), token_response=data.get('token_response', None), scopes=data.get('scopes', None), token_info_uri=data.get('token_info_uri', None))
        retval.invalid = data['invalid']
        return retval

    @property
    def access_token_expired(self):
        """True if the credential is expired or invalid.

        If the token_expiry isn't set, we assume the token doesn't expire.
        """
        if self.invalid:
            return True
        if not self.token_expiry:
            return False
        now = _UTCNOW()
        if now >= self.token_expiry:
            logger.info('access_token is expired. Now: %s, token_expiry: %s', now, self.token_expiry)
            return True
        return False

    def get_access_token(self, http=None):
        """Return the access token and its expiration information.

        If the token does not exist, get one.
        If the token expired, refresh it.
        """
        if not self.access_token or self.access_token_expired:
            if not http:
                http = transport.get_http_object()
            self.refresh(http)
        return AccessTokenInfo(access_token=self.access_token, expires_in=self._expires_in())

    def set_store(self, store):
        """Set the Storage for the credential.

        Args:
            store: Storage, an implementation of Storage object.
                   This is needed to store the latest access_token if it
                   has expired and been refreshed. This implementation uses
                   locking to check for updates before updating the
                   access_token.
        """
        self.store = store

    def _expires_in(self):
        """Return the number of seconds until this token expires.

        If token_expiry is in the past, this method will return 0, meaning the
        token has already expired.

        If token_expiry is None, this method will return None. Note that
        returning 0 in such a case would not be fair: the token may still be
        valid; we just don't know anything about it.
        """
        if self.token_expiry:
            now = _UTCNOW()
            if self.token_expiry > now:
                time_delta = self.token_expiry - now
                return time_delta.days * 86400 + time_delta.seconds
            else:
                return 0

    def _updateFromCredential(self, other):
        """Update this Credential from another instance."""
        self.__dict__.update(other.__getstate__())

    def __getstate__(self):
        """Trim the state down to something that can be pickled."""
        d = copy.copy(self.__dict__)
        del d['store']
        return d

    def __setstate__(self, state):
        """Reconstitute the state of the object from being pickled."""
        self.__dict__.update(state)
        self.store = None

    def _generate_refresh_request_body(self):
        """Generate the body that will be used in the refresh request."""
        body = urllib.parse.urlencode({'grant_type': 'refresh_token', 'client_id': self.client_id, 'client_secret': self.client_secret, 'refresh_token': self.refresh_token})
        return body

    def _generate_refresh_request_headers(self):
        """Generate the headers that will be used in the refresh request."""
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        if self.user_agent is not None:
            headers['user-agent'] = self.user_agent
        return headers

    def _refresh(self, http):
        """Refreshes the access_token.

        This method first checks by reading the Storage object if available.
        If a refresh is still needed, it holds the Storage lock until the
        refresh is completed.

        Args:
            http: an object to be used to make HTTP requests.

        Raises:
            HttpAccessTokenRefreshError: When the refresh fails.
        """
        if not self.store:
            self._do_refresh_request(http)
        else:
            self.store.acquire_lock()
            try:
                new_cred = self.store.locked_get()
                if new_cred and (not new_cred.invalid) and (new_cred.access_token != self.access_token) and (not new_cred.access_token_expired):
                    logger.info('Updated access_token read from Storage')
                    self._updateFromCredential(new_cred)
                else:
                    self._do_refresh_request(http)
            finally:
                self.store.release_lock()

    def _do_refresh_request(self, http):
        """Refresh the access_token using the refresh_token.

        Args:
            http: an object to be used to make HTTP requests.

        Raises:
            HttpAccessTokenRefreshError: When the refresh fails.
        """
        body = self._generate_refresh_request_body()
        headers = self._generate_refresh_request_headers()
        logger.info('Refreshing access_token')
        resp, content = transport.request(http, self.token_uri, method='POST', body=body, headers=headers)
        content = _helpers._from_bytes(content)
        if resp.status == http_client.OK:
            d = json.loads(content)
            self.token_response = d
            self.access_token = d['access_token']
            self.refresh_token = d.get('refresh_token', self.refresh_token)
            if 'expires_in' in d:
                delta = datetime.timedelta(seconds=int(d['expires_in']))
                self.token_expiry = delta + _UTCNOW()
            else:
                self.token_expiry = None
            if 'id_token' in d:
                self.id_token = _extract_id_token(d['id_token'])
                self.id_token_jwt = d['id_token']
            else:
                self.id_token = None
                self.id_token_jwt = None
            self.invalid = False
            if self.store:
                self.store.locked_put(self)
        else:
            logger.info('Failed to retrieve access token: %s', content)
            error_msg = 'Invalid response {0}.'.format(resp.status)
            try:
                d = json.loads(content)
                if 'error' in d:
                    error_msg = d['error']
                    if 'error_description' in d:
                        error_msg += ': ' + d['error_description']
                    self.invalid = True
                    if self.store is not None:
                        self.store.locked_put(self)
            except (TypeError, ValueError):
                pass
            raise HttpAccessTokenRefreshError(error_msg, status=resp.status)

    def _revoke(self, http):
        """Revokes this credential and deletes the stored copy (if it exists).

        Args:
            http: an object to be used to make HTTP requests.
        """
        self._do_revoke(http, self.refresh_token or self.access_token)

    def _do_revoke(self, http, token):
        """Revokes this credential and deletes the stored copy (if it exists).

        Args:
            http: an object to be used to make HTTP requests.
            token: A string used as the token to be revoked. Can be either an
                   access_token or refresh_token.

        Raises:
            TokenRevokeError: If the revoke request does not return with a
                              200 OK.
        """
        logger.info('Revoking token')
        query_params = {'token': token}
        token_revoke_uri = _helpers.update_query_params(self.revoke_uri, query_params)
        resp, content = transport.request(http, token_revoke_uri)
        if resp.status == http_client.METHOD_NOT_ALLOWED:
            body = urllib.parse.urlencode(query_params)
            resp, content = transport.request(http, token_revoke_uri, method='POST', body=body)
        if resp.status == http_client.OK:
            self.invalid = True
        else:
            error_msg = 'Invalid response {0}.'.format(resp.status)
            try:
                d = json.loads(_helpers._from_bytes(content))
                if 'error' in d:
                    error_msg = d['error']
            except (TypeError, ValueError):
                pass
            raise TokenRevokeError(error_msg)
        if self.store:
            self.store.delete()

    def _retrieve_scopes(self, http):
        """Retrieves the list of authorized scopes from the OAuth2 provider.

        Args:
            http: an object to be used to make HTTP requests.
        """
        self._do_retrieve_scopes(http, self.access_token)

    def _do_retrieve_scopes(self, http, token):
        """Retrieves the list of authorized scopes from the OAuth2 provider.

        Args:
            http: an object to be used to make HTTP requests.
            token: A string used as the token to identify the credentials to
                   the provider.

        Raises:
            Error: When refresh fails, indicating the the access token is
                   invalid.
        """
        logger.info('Refreshing scopes')
        query_params = {'access_token': token, 'fields': 'scope'}
        token_info_uri = _helpers.update_query_params(self.token_info_uri, query_params)
        resp, content = transport.request(http, token_info_uri)
        content = _helpers._from_bytes(content)
        if resp.status == http_client.OK:
            d = json.loads(content)
            self.scopes = set(_helpers.string_to_scopes(d.get('scope', '')))
        else:
            error_msg = 'Invalid response {0}.'.format(resp.status)
            try:
                d = json.loads(content)
                if 'error_description' in d:
                    error_msg = d['error_description']
            except (TypeError, ValueError):
                pass
            raise Error(error_msg)