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
class OAuthMixin(object):
    """Abstract implementation of OAuth 1.0 and 1.0a.

    See `TwitterMixin` below for an example implementation.

    Class attributes:

    * ``_OAUTH_AUTHORIZE_URL``: The service's OAuth authorization url.
    * ``_OAUTH_ACCESS_TOKEN_URL``: The service's OAuth access token url.
    * ``_OAUTH_VERSION``: May be either "1.0" or "1.0a".
    * ``_OAUTH_NO_CALLBACKS``: Set this to True if the service requires
      advance registration of callbacks.

    Subclasses must also override the `_oauth_get_user_future` and
    `_oauth_consumer_token` methods.
    """

    async def authorize_redirect(self, callback_uri: Optional[str]=None, extra_params: Optional[Dict[str, Any]]=None, http_client: Optional[httpclient.AsyncHTTPClient]=None) -> None:
        """Redirects the user to obtain OAuth authorization for this service.

        The ``callback_uri`` may be omitted if you have previously
        registered a callback URI with the third-party service. For
        some services, you must use a previously-registered callback
        URI and cannot specify a callback via this method.

        This method sets a cookie called ``_oauth_request_token`` which is
        subsequently used (and cleared) in `get_authenticated_user` for
        security purposes.

        This method is asynchronous and must be called with ``await``
        or ``yield`` (This is different from other ``auth*_redirect``
        methods defined in this module). It calls
        `.RequestHandler.finish` for you so you should not write any
        other response after it returns.

        .. versionchanged:: 3.1
           Now returns a `.Future` and takes an optional callback, for
           compatibility with `.gen.coroutine`.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           awaitable object instead.

        """
        if callback_uri and getattr(self, '_OAUTH_NO_CALLBACKS', False):
            raise Exception('This service does not support oauth_callback')
        if http_client is None:
            http_client = self.get_auth_http_client()
        assert http_client is not None
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            response = await http_client.fetch(self._oauth_request_token_url(callback_uri=callback_uri, extra_params=extra_params))
        else:
            response = await http_client.fetch(self._oauth_request_token_url())
        url = self._OAUTH_AUTHORIZE_URL
        self._on_request_token(url, callback_uri, response)

    async def get_authenticated_user(self, http_client: Optional[httpclient.AsyncHTTPClient]=None) -> Dict[str, Any]:
        """Gets the OAuth authorized user and access token.

        This method should be called from the handler for your
        OAuth callback URL to complete the registration process. We run the
        callback with the authenticated user dictionary.  This dictionary
        will contain an ``access_key`` which can be used to make authorized
        requests to this service on behalf of the user.  The dictionary will
        also contain other fields such as ``name``, depending on the service
        used.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           awaitable object instead.
        """
        handler = cast(RequestHandler, self)
        request_key = escape.utf8(handler.get_argument('oauth_token'))
        oauth_verifier = handler.get_argument('oauth_verifier', None)
        request_cookie = handler.get_cookie('_oauth_request_token')
        if not request_cookie:
            raise AuthError('Missing OAuth request token cookie')
        handler.clear_cookie('_oauth_request_token')
        cookie_key, cookie_secret = [base64.b64decode(escape.utf8(i)) for i in request_cookie.split('|')]
        if cookie_key != request_key:
            raise AuthError('Request token does not match cookie')
        token = dict(key=cookie_key, secret=cookie_secret)
        if oauth_verifier:
            token['verifier'] = oauth_verifier
        if http_client is None:
            http_client = self.get_auth_http_client()
        assert http_client is not None
        response = await http_client.fetch(self._oauth_access_token_url(token))
        access_token = _oauth_parse_response(response.body)
        user = await self._oauth_get_user_future(access_token)
        if not user:
            raise AuthError('Error getting user')
        user['access_token'] = access_token
        return user

    def _oauth_request_token_url(self, callback_uri: Optional[str]=None, extra_params: Optional[Dict[str, Any]]=None) -> str:
        handler = cast(RequestHandler, self)
        consumer_token = self._oauth_consumer_token()
        url = self._OAUTH_REQUEST_TOKEN_URL
        args = dict(oauth_consumer_key=escape.to_basestring(consumer_token['key']), oauth_signature_method='HMAC-SHA1', oauth_timestamp=str(int(time.time())), oauth_nonce=escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)), oauth_version='1.0')
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            if callback_uri == 'oob':
                args['oauth_callback'] = 'oob'
            elif callback_uri:
                args['oauth_callback'] = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
            if extra_params:
                args.update(extra_params)
            signature = _oauth10a_signature(consumer_token, 'GET', url, args)
        else:
            signature = _oauth_signature(consumer_token, 'GET', url, args)
        args['oauth_signature'] = signature
        return url + '?' + urllib.parse.urlencode(args)

    def _on_request_token(self, authorize_url: str, callback_uri: Optional[str], response: httpclient.HTTPResponse) -> None:
        handler = cast(RequestHandler, self)
        request_token = _oauth_parse_response(response.body)
        data = base64.b64encode(escape.utf8(request_token['key'])) + b'|' + base64.b64encode(escape.utf8(request_token['secret']))
        handler.set_cookie('_oauth_request_token', data)
        args = dict(oauth_token=request_token['key'])
        if callback_uri == 'oob':
            handler.finish(authorize_url + '?' + urllib.parse.urlencode(args))
            return
        elif callback_uri:
            args['oauth_callback'] = urllib.parse.urljoin(handler.request.full_url(), callback_uri)
        handler.redirect(authorize_url + '?' + urllib.parse.urlencode(args))

    def _oauth_access_token_url(self, request_token: Dict[str, Any]) -> str:
        consumer_token = self._oauth_consumer_token()
        url = self._OAUTH_ACCESS_TOKEN_URL
        args = dict(oauth_consumer_key=escape.to_basestring(consumer_token['key']), oauth_token=escape.to_basestring(request_token['key']), oauth_signature_method='HMAC-SHA1', oauth_timestamp=str(int(time.time())), oauth_nonce=escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)), oauth_version='1.0')
        if 'verifier' in request_token:
            args['oauth_verifier'] = request_token['verifier']
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            signature = _oauth10a_signature(consumer_token, 'GET', url, args, request_token)
        else:
            signature = _oauth_signature(consumer_token, 'GET', url, args, request_token)
        args['oauth_signature'] = signature
        return url + '?' + urllib.parse.urlencode(args)

    def _oauth_consumer_token(self) -> Dict[str, Any]:
        """Subclasses must override this to return their OAuth consumer keys.

        The return value should be a `dict` with keys ``key`` and ``secret``.
        """
        raise NotImplementedError()

    async def _oauth_get_user_future(self, access_token: Dict[str, Any]) -> Dict[str, Any]:
        """Subclasses must override this to get basic information about the
        user.

        Should be a coroutine whose result is a dictionary
        containing information about the user, which may have been
        retrieved by using ``access_token`` to make a request to the
        service.

        The access token will be added to the returned dictionary to make
        the result of `get_authenticated_user`.

        .. versionchanged:: 5.1

           Subclasses may also define this method with ``async def``.

        .. versionchanged:: 6.0

           A synchronous fallback to ``_oauth_get_user`` was removed.
        """
        raise NotImplementedError()

    def _oauth_request_parameters(self, url: str, access_token: Dict[str, Any], parameters: Dict[str, Any]={}, method: str='GET') -> Dict[str, Any]:
        """Returns the OAuth parameters as a dict for the given request.

        parameters should include all POST arguments and query string arguments
        that will be sent with the request.
        """
        consumer_token = self._oauth_consumer_token()
        base_args = dict(oauth_consumer_key=escape.to_basestring(consumer_token['key']), oauth_token=escape.to_basestring(access_token['key']), oauth_signature_method='HMAC-SHA1', oauth_timestamp=str(int(time.time())), oauth_nonce=escape.to_basestring(binascii.b2a_hex(uuid.uuid4().bytes)), oauth_version='1.0')
        args = {}
        args.update(base_args)
        args.update(parameters)
        if getattr(self, '_OAUTH_VERSION', '1.0a') == '1.0a':
            signature = _oauth10a_signature(consumer_token, method, url, args, access_token)
        else:
            signature = _oauth_signature(consumer_token, method, url, args, access_token)
        base_args['oauth_signature'] = escape.to_basestring(signature)
        return base_args

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        """Returns the `.AsyncHTTPClient` instance to be used for auth requests.

        May be overridden by subclasses to use an HTTP client other than
        the default.
        """
        return httpclient.AsyncHTTPClient()