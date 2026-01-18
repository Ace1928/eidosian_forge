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
class FacebookGraphMixin(OAuth2Mixin):
    """Facebook authentication using the new Graph API and OAuth2."""
    _OAUTH_ACCESS_TOKEN_URL = 'https://graph.facebook.com/oauth/access_token?'
    _OAUTH_AUTHORIZE_URL = 'https://www.facebook.com/dialog/oauth?'
    _OAUTH_NO_CALLBACKS = False
    _FACEBOOK_BASE_URL = 'https://graph.facebook.com'

    async def get_authenticated_user(self, redirect_uri: str, client_id: str, client_secret: str, code: str, extra_fields: Optional[Dict[str, Any]]=None) -> Optional[Dict[str, Any]]:
        """Handles the login for the Facebook user, returning a user object.

        Example usage:

        .. testcode::

            class FacebookGraphLoginHandler(tornado.web.RequestHandler,
                                            tornado.auth.FacebookGraphMixin):
              async def get(self):
                redirect_uri = urllib.parse.urljoin(
                    self.application.settings['redirect_base_uri'],
                    self.reverse_url('facebook_oauth'))
                if self.get_argument("code", False):
                    user = await self.get_authenticated_user(
                        redirect_uri=redirect_uri,
                        client_id=self.settings["facebook_api_key"],
                        client_secret=self.settings["facebook_secret"],
                        code=self.get_argument("code"))
                    # Save the user with e.g. set_signed_cookie
                else:
                    self.authorize_redirect(
                        redirect_uri=redirect_uri,
                        client_id=self.settings["facebook_api_key"],
                        extra_params={"scope": "user_posts"})

        .. testoutput::
           :hide:

        This method returns a dictionary which may contain the following fields:

        * ``access_token``, a string which may be passed to `facebook_request`
        * ``session_expires``, an integer encoded as a string representing
          the time until the access token expires in seconds. This field should
          be used like ``int(user['session_expires'])``; in a future version of
          Tornado it will change from a string to an integer.
        * ``id``, ``name``, ``first_name``, ``last_name``, ``locale``, ``picture``,
          ``link``, plus any fields named in the ``extra_fields`` argument. These
          fields are copied from the Facebook graph API
          `user object <https://developers.facebook.com/docs/graph-api/reference/user>`_

        .. versionchanged:: 4.5
           The ``session_expires`` field was updated to support changes made to the
           Facebook API in March 2017.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned awaitable object instead.
        """
        http = self.get_auth_http_client()
        args = {'redirect_uri': redirect_uri, 'code': code, 'client_id': client_id, 'client_secret': client_secret}
        fields = set(['id', 'name', 'first_name', 'last_name', 'locale', 'picture', 'link'])
        if extra_fields:
            fields.update(extra_fields)
        response = await http.fetch(self._oauth_request_token_url(**args))
        args = escape.json_decode(response.body)
        session = {'access_token': args.get('access_token'), 'expires_in': args.get('expires_in')}
        assert session['access_token'] is not None
        user = await self.facebook_request(path='/me', access_token=session['access_token'], appsecret_proof=hmac.new(key=client_secret.encode('utf8'), msg=session['access_token'].encode('utf8'), digestmod=hashlib.sha256).hexdigest(), fields=','.join(fields))
        if user is None:
            return None
        fieldmap = {}
        for field in fields:
            fieldmap[field] = user.get(field)
        fieldmap.update({'access_token': session['access_token'], 'session_expires': str(session.get('expires_in'))})
        return fieldmap

    async def facebook_request(self, path: str, access_token: Optional[str]=None, post_args: Optional[Dict[str, Any]]=None, **args: Any) -> Any:
        """Fetches the given relative API path, e.g., "/btaylor/picture"

        If the request is a POST, ``post_args`` should be provided. Query
        string arguments should be given as keyword arguments.

        An introduction to the Facebook Graph API can be found at
        http://developers.facebook.com/docs/api

        Many methods require an OAuth access token which you can
        obtain through `~OAuth2Mixin.authorize_redirect` and
        `get_authenticated_user`. The user returned through that
        process includes an ``access_token`` attribute that can be
        used to make authenticated requests via this method.

        Example usage:

        .. testcode::

            class MainHandler(tornado.web.RequestHandler,
                              tornado.auth.FacebookGraphMixin):
                @tornado.web.authenticated
                async def get(self):
                    new_entry = await self.facebook_request(
                        "/me/feed",
                        post_args={"message": "I am posting from my Tornado application!"},
                        access_token=self.current_user["access_token"])

                    if not new_entry:
                        # Call failed; perhaps missing permission?
                        self.authorize_redirect()
                        return
                    self.finish("Posted a message!")

        .. testoutput::
           :hide:

        The given path is relative to ``self._FACEBOOK_BASE_URL``,
        by default "https://graph.facebook.com".

        This method is a wrapper around `OAuth2Mixin.oauth2_request`;
        the only difference is that this method takes a relative path,
        while ``oauth2_request`` takes a complete url.

        .. versionchanged:: 3.1
           Added the ability to override ``self._FACEBOOK_BASE_URL``.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned awaitable object instead.
        """
        url = self._FACEBOOK_BASE_URL + path
        return await self.oauth2_request(url, access_token=access_token, post_args=post_args, **args)