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
class OAuth2DecoratorFromClientSecrets(OAuth2Decorator):
    """An OAuth2Decorator that builds from a clientsecrets file.

    Uses a clientsecrets file as the source for all the information when
    constructing an OAuth2Decorator.

    ::

        decorator = OAuth2DecoratorFromClientSecrets(
            os.path.join(os.path.dirname(__file__), 'client_secrets.json')
            scope='https://www.googleapis.com/auth/plus')

        class MainHandler(webapp.RequestHandler):
            @decorator.oauth_required
            def get(self):
                http = decorator.http()
                # http is authorized with the user's Credentials and can be
                # used in API calls

    """

    @_helpers.positional(3)
    def __init__(self, filename, scope, message=None, cache=None, **kwargs):
        """Constructor

        Args:
            filename: string, File name of client secrets.
            scope: string or iterable of strings, scope(s) of the credentials
                   being requested.
            message: string, A friendly string to display to the user if the
                     clientsecrets file is missing or invalid. The message may
                     contain HTML and will be presented on the web interface
                     for any method that uses the decorator.
            cache: An optional cache service client that implements get() and
                   set()
            methods. See clientsecrets.loadfile() for details.
            **kwargs: dict, Keyword arguments are passed along as kwargs to
                      the OAuth2WebServerFlow constructor.
        """
        client_type, client_info = clientsecrets.loadfile(filename, cache=cache)
        if client_type not in (clientsecrets.TYPE_WEB, clientsecrets.TYPE_INSTALLED):
            raise clientsecrets.InvalidClientSecretsError("OAuth2Decorator doesn't support this OAuth 2.0 flow.")
        constructor_kwargs = dict(kwargs)
        constructor_kwargs.update({'auth_uri': client_info['auth_uri'], 'token_uri': client_info['token_uri'], 'message': message})
        revoke_uri = client_info.get('revoke_uri')
        if revoke_uri is not None:
            constructor_kwargs['revoke_uri'] = revoke_uri
        super(OAuth2DecoratorFromClientSecrets, self).__init__(client_info['client_id'], client_info['client_secret'], scope, **constructor_kwargs)
        if message is not None:
            self._message = message
        else:
            self._message = 'Please configure your application for OAuth 2.0.'