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
class OAuth2Decorator(object):
    """Utility for making OAuth 2.0 easier.

    Instantiate and then use with oauth_required or oauth_aware
    as decorators on webapp.RequestHandler methods.

    ::

        decorator = OAuth2Decorator(
            client_id='837...ent.com',
            client_secret='Qh...wwI',
            scope='https://www.googleapis.com/auth/plus')

        class MainHandler(webapp.RequestHandler):
            @decorator.oauth_required
            def get(self):
                http = decorator.http()
                # http is authorized with the user's Credentials and can be
                # used in API calls

    """

    def set_credentials(self, credentials):
        self._tls.credentials = credentials

    def get_credentials(self):
        """A thread local Credentials object.

        Returns:
            A client.Credentials object, or None if credentials hasn't been set
            in this thread yet, which may happen when calling has_credentials
            inside oauth_aware.
        """
        return getattr(self._tls, 'credentials', None)
    credentials = property(get_credentials, set_credentials)

    def set_flow(self, flow):
        self._tls.flow = flow

    def get_flow(self):
        """A thread local Flow object.

        Returns:
            A credentials.Flow object, or None if the flow hasn't been set in
            this thread yet, which happens in _create_flow() since Flows are
            created lazily.
        """
        return getattr(self._tls, 'flow', None)
    flow = property(get_flow, set_flow)

    @_helpers.positional(4)
    def __init__(self, client_id, client_secret, scope, auth_uri=oauth2client.GOOGLE_AUTH_URI, token_uri=oauth2client.GOOGLE_TOKEN_URI, revoke_uri=oauth2client.GOOGLE_REVOKE_URI, user_agent=None, message=None, callback_path='/oauth2callback', token_response_param=None, _storage_class=StorageByKeyName, _credentials_class=CredentialsModel, _credentials_property_name='credentials', **kwargs):
        """Constructor for OAuth2Decorator

        Args:
            client_id: string, client identifier.
            client_secret: string client secret.
            scope: string or iterable of strings, scope(s) of the credentials
                   being requested.
            auth_uri: string, URI for authorization endpoint. For convenience
                      defaults to Google's endpoints but any OAuth 2.0 provider
                      can be used.
            token_uri: string, URI for token endpoint. For convenience defaults
                       to Google's endpoints but any OAuth 2.0 provider can be
                       used.
            revoke_uri: string, URI for revoke endpoint. For convenience
                        defaults to Google's endpoints but any OAuth 2.0
                        provider can be used.
            user_agent: string, User agent of your application, default to
                        None.
            message: Message to display if there are problems with the
                     OAuth 2.0 configuration. The message may contain HTML and
                     will be presented on the web interface for any method that
                     uses the decorator.
            callback_path: string, The absolute path to use as the callback
                           URI. Note that this must match up with the URI given
                           when registering the application in the APIs
                           Console.
            token_response_param: string. If provided, the full JSON response
                                  to the access token request will be encoded
                                  and included in this query parameter in the
                                  callback URI. This is useful with providers
                                  (e.g. wordpress.com) that include extra
                                  fields that the client may want.
            _storage_class: "Protected" keyword argument not typically provided
                            to this constructor. A storage class to aid in
                            storing a Credentials object for a user in the
                            datastore. Defaults to StorageByKeyName.
            _credentials_class: "Protected" keyword argument not typically
                                provided to this constructor. A db or ndb Model
                                class to hold credentials. Defaults to
                                CredentialsModel.
            _credentials_property_name: "Protected" keyword argument not
                                        typically provided to this constructor.
                                        A string indicating the name of the
                                        field on the _credentials_class where a
                                        Credentials object will be stored.
                                        Defaults to 'credentials'.
            **kwargs: dict, Keyword arguments are passed along as kwargs to
                      the OAuth2WebServerFlow constructor.
        """
        self._tls = threading.local()
        self.flow = None
        self.credentials = None
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = _helpers.scopes_to_string(scope)
        self._auth_uri = auth_uri
        self._token_uri = token_uri
        self._revoke_uri = revoke_uri
        self._user_agent = user_agent
        self._kwargs = kwargs
        self._message = message
        self._in_error = False
        self._callback_path = callback_path
        self._token_response_param = token_response_param
        self._storage_class = _storage_class
        self._credentials_class = _credentials_class
        self._credentials_property_name = _credentials_property_name

    def _display_error_message(self, request_handler):
        request_handler.response.out.write('<html><body>')
        request_handler.response.out.write(_safe_html(self._message))
        request_handler.response.out.write('</body></html>')

    def oauth_required(self, method):
        """Decorator that starts the OAuth 2.0 dance.

        Starts the OAuth dance for the logged in user if they haven't already
        granted access for this application.

        Args:
            method: callable, to be decorated method of a webapp.RequestHandler
                    instance.
        """

        def check_oauth(request_handler, *args, **kwargs):
            if self._in_error:
                self._display_error_message(request_handler)
                return
            user = users.get_current_user()
            if not user:
                request_handler.redirect(users.create_login_url(request_handler.request.uri))
                return
            self._create_flow(request_handler)
            self.flow.params['state'] = _build_state_value(request_handler, user)
            self.credentials = self._storage_class(self._credentials_class, None, self._credentials_property_name, user=user).get()
            if not self.has_credentials():
                return request_handler.redirect(self.authorize_url())
            try:
                resp = method(request_handler, *args, **kwargs)
            except client.AccessTokenRefreshError:
                return request_handler.redirect(self.authorize_url())
            finally:
                self.credentials = None
            return resp
        return check_oauth

    def _create_flow(self, request_handler):
        """Create the Flow object.

        The Flow is calculated lazily since we don't know where this app is
        running until it receives a request, at which point redirect_uri can be
        calculated and then the Flow object can be constructed.

        Args:
            request_handler: webapp.RequestHandler, the request handler.
        """
        if self.flow is None:
            redirect_uri = request_handler.request.relative_url(self._callback_path)
            self.flow = client.OAuth2WebServerFlow(self._client_id, self._client_secret, self._scope, redirect_uri=redirect_uri, user_agent=self._user_agent, auth_uri=self._auth_uri, token_uri=self._token_uri, revoke_uri=self._revoke_uri, **self._kwargs)

    def oauth_aware(self, method):
        """Decorator that sets up for OAuth 2.0 dance, but doesn't do it.

        Does all the setup for the OAuth dance, but doesn't initiate it.
        This decorator is useful if you want to create a page that knows
        whether or not the user has granted access to this application.
        From within a method decorated with @oauth_aware the has_credentials()
        and authorize_url() methods can be called.

        Args:
            method: callable, to be decorated method of a webapp.RequestHandler
                    instance.
        """

        def setup_oauth(request_handler, *args, **kwargs):
            if self._in_error:
                self._display_error_message(request_handler)
                return
            user = users.get_current_user()
            if not user:
                request_handler.redirect(users.create_login_url(request_handler.request.uri))
                return
            self._create_flow(request_handler)
            self.flow.params['state'] = _build_state_value(request_handler, user)
            self.credentials = self._storage_class(self._credentials_class, None, self._credentials_property_name, user=user).get()
            try:
                resp = method(request_handler, *args, **kwargs)
            finally:
                self.credentials = None
            return resp
        return setup_oauth

    def has_credentials(self):
        """True if for the logged in user there are valid access Credentials.

        Must only be called from with a webapp.RequestHandler subclassed method
        that had been decorated with either @oauth_required or @oauth_aware.
        """
        return self.credentials is not None and (not self.credentials.invalid)

    def authorize_url(self):
        """Returns the URL to start the OAuth dance.

        Must only be called from with a webapp.RequestHandler subclassed method
        that had been decorated with either @oauth_required or @oauth_aware.
        """
        url = self.flow.step1_get_authorize_url()
        return str(url)

    def http(self, *args, **kwargs):
        """Returns an authorized http instance.

        Must only be called from within an @oauth_required decorated method, or
        from within an @oauth_aware decorated method where has_credentials()
        returns True.

        Args:
            *args: Positional arguments passed to httplib2.Http constructor.
            **kwargs: Positional arguments passed to httplib2.Http constructor.
        """
        return self.credentials.authorize(transport.get_http_object(*args, **kwargs))

    @property
    def callback_path(self):
        """The absolute path where the callback will occur.

        Note this is the absolute path, not the absolute URI, that will be
        calculated by the decorator at runtime. See callback_handler() for how
        this should be used.

        Returns:
            The callback path as a string.
        """
        return self._callback_path

    def callback_handler(self):
        """RequestHandler for the OAuth 2.0 redirect callback.

        Usage::

            app = webapp.WSGIApplication([
                ('/index', MyIndexHandler),
                ...,
                (decorator.callback_path, decorator.callback_handler())
            ])

        Returns:
            A webapp.RequestHandler that handles the redirect back from the
            server during the OAuth 2.0 dance.
        """
        decorator = self

        class OAuth2Handler(webapp.RequestHandler):
            """Handler for the redirect_uri of the OAuth 2.0 dance."""

            @login_required
            def get(self):
                error = self.request.get('error')
                if error:
                    errormsg = self.request.get('error_description', error)
                    self.response.out.write('The authorization request failed: {0}'.format(_safe_html(errormsg)))
                else:
                    user = users.get_current_user()
                    decorator._create_flow(self)
                    credentials = decorator.flow.step2_exchange(self.request.params)
                    decorator._storage_class(decorator._credentials_class, None, decorator._credentials_property_name, user=user).put(credentials)
                    redirect_uri = _parse_state_value(str(self.request.get('state')), user)
                    if redirect_uri is None:
                        self.response.out.write('The authorization request failed')
                        return
                    if decorator._token_response_param and credentials.token_response:
                        resp_json = json.dumps(credentials.token_response)
                        redirect_uri = _helpers._add_query_parameter(redirect_uri, decorator._token_response_param, resp_json)
                    self.redirect(redirect_uri)
        return OAuth2Handler

    def callback_application(self):
        """WSGI application for handling the OAuth 2.0 redirect callback.

        If you need finer grained control use `callback_handler` which returns
        just the webapp.RequestHandler.

        Returns:
            A webapp.WSGIApplication that handles the redirect back from the
            server during the OAuth 2.0 dance.
        """
        return webapp.WSGIApplication([(self.callback_path, self.callback_handler())])