import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional
class GoogleOAuth2Mixin(OAuth2Mixin):
    """Google authentication using OAuth2.

    In order to use, register your application with Google and copy the
    relevant parameters to your application settings.

    * Go to the Google Dev Console at http://console.developers.google.com
    * Select a project, or create a new one.
    * Depending on permissions required, you may need to set your app to
      "testing" mode and add your account as a test user, or go through
      a verfication process. You may also need to use the "Enable
      APIs and Services" command to enable specific services.
    * In the sidebar on the left, select Credentials.
    * Click CREATE CREDENTIALS and click OAuth client ID.
    * Under Application type, select Web application.
    * Name OAuth 2.0 client and click Create.
    * Copy the "Client secret" and "Client ID" to the application settings as
      ``{"google_oauth": {"key": CLIENT_ID, "secret": CLIENT_SECRET}}``
    * You must register the ``redirect_uri`` you plan to use with this class
      on the Credentials page.

    .. versionadded:: 3.2
    """
    _OAUTH_AUTHORIZE_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
    _OAUTH_ACCESS_TOKEN_URL = 'https://www.googleapis.com/oauth2/v4/token'
    _OAUTH_USERINFO_URL = 'https://www.googleapis.com/oauth2/v1/userinfo'
    _OAUTH_NO_CALLBACKS = False
    _OAUTH_SETTINGS_KEY = 'google_oauth'

    def get_google_oauth_settings(self) -> Dict[str, str]:
        """Return the Google OAuth 2.0 credentials that you created with
        [Google Cloud
        Platform](https://console.cloud.google.com/apis/credentials). The dict
        format is::

            {
                "key": "your_client_id", "secret": "your_client_secret"
            }

        If your credentials are stored differently (e.g. in a db) you can
        override this method for custom provision.
        """
        handler = cast(RequestHandler, self)
        return handler.settings[self._OAUTH_SETTINGS_KEY]

    async def get_authenticated_user(self, redirect_uri: str, code: str, client_id: Optional[str]=None, client_secret: Optional[str]=None) -> Dict[str, Any]:
        """Handles the login for the Google user, returning an access token.

        The result is a dictionary containing an ``access_token`` field
        ([among others](https://developers.google.com/identity/protocols/OAuth2WebServer#handlingtheresponse)).
        Unlike other ``get_authenticated_user`` methods in this package,
        this method does not return any additional information about the user.
        The returned access token can be used with `OAuth2Mixin.oauth2_request`
        to request additional information (perhaps from
        ``https://www.googleapis.com/oauth2/v2/userinfo``)

        Example usage:

        .. testsetup::

            import urllib

        .. testcode::

            class GoogleOAuth2LoginHandler(tornado.web.RequestHandler,
                                           tornado.auth.GoogleOAuth2Mixin):
                async def get(self):
                    # Google requires an exact match for redirect_uri, so it's
                    # best to get it from your app configuration instead of from
                    # self.request.full_uri().
                    redirect_uri = urllib.parse.urljoin(self.application.settings['redirect_base_uri'],
                        self.reverse_url('google_oauth'))
                    async def get(self):
                        if self.get_argument('code', False):
                            access = await self.get_authenticated_user(
                                redirect_uri=redirect_uri,
                                code=self.get_argument('code'))
                            user = await self.oauth2_request(
                                "https://www.googleapis.com/oauth2/v1/userinfo",
                                access_token=access["access_token"])
                            # Save the user and access token. For example:
                            user_cookie = dict(id=user["id"], access_token=access["access_token"])
                            self.set_signed_cookie("user", json.dumps(user_cookie))
                            self.redirect("/")
                        else:
                            self.authorize_redirect(
                                redirect_uri=redirect_uri,
                                client_id=self.get_google_oauth_settings()['key'],
                                scope=['profile', 'email'],
                                response_type='code',
                                extra_params={'approval_prompt': 'auto'})

        .. testoutput::
           :hide:

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned awaitable object instead.
        """
        if client_id is None or client_secret is None:
            settings = self.get_google_oauth_settings()
            if client_id is None:
                client_id = settings['key']
            if client_secret is None:
                client_secret = settings['secret']
        http = self.get_auth_http_client()
        body = urllib.parse.urlencode({'redirect_uri': redirect_uri, 'code': code, 'client_id': client_id, 'client_secret': client_secret, 'grant_type': 'authorization_code'})
        response = await http.fetch(self._OAUTH_ACCESS_TOKEN_URL, method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded'}, body=body)
        return escape.json_decode(response.body)