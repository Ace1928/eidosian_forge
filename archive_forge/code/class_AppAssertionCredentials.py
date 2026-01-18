import cgi
import json
import logging
import os
import pickle
import threading
from google.appengine.api import app_identity
from google.appengine.api import memcache
from google.appengine.api import users
from google.appengine.ext import db
from google.appengine.ext.webapp.util import login_required
import webapp2 as webapp
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import xsrfutil
class AppAssertionCredentials(client.AssertionCredentials):
    """Credentials object for App Engine Assertion Grants

    This object will allow an App Engine application to identify itself to
    Google and other OAuth 2.0 servers that can verify assertions. It can be
    used for the purpose of accessing data stored under an account assigned to
    the App Engine application itself.

    This credential does not require a flow to instantiate because it
    represents a two legged flow, and therefore has all of the required
    information to generate and refresh its own access tokens.
    """

    @_helpers.positional(2)
    def __init__(self, scope, **kwargs):
        """Constructor for AppAssertionCredentials

        Args:
            scope: string or iterable of strings, scope(s) of the credentials
                   being requested.
            **kwargs: optional keyword args, including:
            service_account_id: service account id of the application. If None
                                or unspecified, the default service account for
                                the app is used.
        """
        self.scope = _helpers.scopes_to_string(scope)
        self._kwargs = kwargs
        self.service_account_id = kwargs.get('service_account_id', None)
        self._service_account_email = None
        super(AppAssertionCredentials, self).__init__(None)

    @classmethod
    def from_json(cls, json_data):
        data = json.loads(json_data)
        return AppAssertionCredentials(data['scope'])

    def _refresh(self, http):
        """Refreshes the access token.

        Since the underlying App Engine app_identity implementation does its
        own caching we can skip all the storage hoops and just to a refresh
        using the API.

        Args:
            http: unused HTTP object

        Raises:
            AccessTokenRefreshError: When the refresh fails.
        """
        try:
            scopes = self.scope.split()
            token, _ = app_identity.get_access_token(scopes, service_account_id=self.service_account_id)
        except app_identity.Error as e:
            raise client.AccessTokenRefreshError(str(e))
        self.access_token = token

    @property
    def serialization_data(self):
        raise NotImplementedError('Cannot serialize credentials for Google App Engine.')

    def create_scoped_required(self):
        return not self.scope

    def create_scoped(self, scopes):
        return AppAssertionCredentials(scopes, **self._kwargs)

    def sign_blob(self, blob):
        """Cryptographically sign a blob (of bytes).

        Implements abstract method
        :meth:`oauth2client.client.AssertionCredentials.sign_blob`.

        Args:
            blob: bytes, Message to be signed.

        Returns:
            tuple, A pair of the private key ID used to sign the blob and
            the signed contents.
        """
        return app_identity.sign_blob(blob)

    @property
    def service_account_email(self):
        """Get the email for the current service account.

        Returns:
            string, The email associated with the Google App Engine
            service account.
        """
        if self._service_account_email is None:
            self._service_account_email = app_identity.get_service_account_name()
        return self._service_account_email