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
class TwitterMixin(OAuthMixin):
    """Twitter OAuth authentication.

    To authenticate with Twitter, register your application with
    Twitter at http://twitter.com/apps. Then copy your Consumer Key
    and Consumer Secret to the application
    `~tornado.web.Application.settings` ``twitter_consumer_key`` and
    ``twitter_consumer_secret``. Use this mixin on the handler for the
    URL you registered as your application's callback URL.

    When your application is set up, you can use this mixin like this
    to authenticate the user with Twitter and get access to their stream:

    .. testcode::

        class TwitterLoginHandler(tornado.web.RequestHandler,
                                  tornado.auth.TwitterMixin):
            async def get(self):
                if self.get_argument("oauth_token", None):
                    user = await self.get_authenticated_user()
                    # Save the user using e.g. set_signed_cookie()
                else:
                    await self.authorize_redirect()

    .. testoutput::
       :hide:

    The user object returned by `~OAuthMixin.get_authenticated_user`
    includes the attributes ``username``, ``name``, ``access_token``,
    and all of the custom Twitter user attributes described at
    https://dev.twitter.com/docs/api/1.1/get/users/show

    .. deprecated:: 6.3
       This class refers to version 1.1 of the Twitter API, which has been
       deprecated by Twitter. Since Twitter has begun to limit access to its
       API, this class will no longer be updated and will be removed in the
       future.
    """
    _OAUTH_REQUEST_TOKEN_URL = 'https://api.twitter.com/oauth/request_token'
    _OAUTH_ACCESS_TOKEN_URL = 'https://api.twitter.com/oauth/access_token'
    _OAUTH_AUTHORIZE_URL = 'https://api.twitter.com/oauth/authorize'
    _OAUTH_AUTHENTICATE_URL = 'https://api.twitter.com/oauth/authenticate'
    _OAUTH_NO_CALLBACKS = False
    _TWITTER_BASE_URL = 'https://api.twitter.com/1.1'

    async def authenticate_redirect(self, callback_uri: Optional[str]=None) -> None:
        """Just like `~OAuthMixin.authorize_redirect`, but
        auto-redirects if authorized.

        This is generally the right interface to use if you are using
        Twitter for single-sign on.

        .. versionchanged:: 3.1
           Now returns a `.Future` and takes an optional callback, for
           compatibility with `.gen.coroutine`.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           awaitable object instead.
        """
        http = self.get_auth_http_client()
        response = await http.fetch(self._oauth_request_token_url(callback_uri=callback_uri))
        self._on_request_token(self._OAUTH_AUTHENTICATE_URL, None, response)

    async def twitter_request(self, path: str, access_token: Dict[str, Any], post_args: Optional[Dict[str, Any]]=None, **args: Any) -> Any:
        """Fetches the given API path, e.g., ``statuses/user_timeline/btaylor``

        The path should not include the format or API version number.
        (we automatically use JSON format and API version 1).

        If the request is a POST, ``post_args`` should be provided. Query
        string arguments should be given as keyword arguments.

        All the Twitter methods are documented at http://dev.twitter.com/

        Many methods require an OAuth access token which you can
        obtain through `~OAuthMixin.authorize_redirect` and
        `~OAuthMixin.get_authenticated_user`. The user returned through that
        process includes an 'access_token' attribute that can be used
        to make authenticated requests via this method. Example
        usage:

        .. testcode::

            class MainHandler(tornado.web.RequestHandler,
                              tornado.auth.TwitterMixin):
                @tornado.web.authenticated
                async def get(self):
                    new_entry = await self.twitter_request(
                        "/statuses/update",
                        post_args={"status": "Testing Tornado Web Server"},
                        access_token=self.current_user["access_token"])
                    if not new_entry:
                        # Call failed; perhaps missing permission?
                        await self.authorize_redirect()
                        return
                    self.finish("Posted a message!")

        .. testoutput::
           :hide:

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           awaitable object instead.
        """
        if path.startswith('http:') or path.startswith('https:'):
            url = path
        else:
            url = self._TWITTER_BASE_URL + path + '.json'
        if access_token:
            all_args = {}
            all_args.update(args)
            all_args.update(post_args or {})
            method = 'POST' if post_args is not None else 'GET'
            oauth = self._oauth_request_parameters(url, access_token, all_args, method=method)
            args.update(oauth)
        if args:
            url += '?' + urllib.parse.urlencode(args)
        http = self.get_auth_http_client()
        if post_args is not None:
            response = await http.fetch(url, method='POST', body=urllib.parse.urlencode(post_args))
        else:
            response = await http.fetch(url)
        return escape.json_decode(response.body)

    def _oauth_consumer_token(self) -> Dict[str, Any]:
        handler = cast(RequestHandler, self)
        handler.require_setting('twitter_consumer_key', 'Twitter OAuth')
        handler.require_setting('twitter_consumer_secret', 'Twitter OAuth')
        return dict(key=handler.settings['twitter_consumer_key'], secret=handler.settings['twitter_consumer_secret'])

    async def _oauth_get_user_future(self, access_token: Dict[str, Any]) -> Dict[str, Any]:
        user = await self.twitter_request('/account/verify_credentials', access_token=access_token)
        if user:
            user['username'] = user['screen_name']
        return user