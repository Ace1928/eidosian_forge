import asyncio
import base64
import codecs
import datetime as dt
import hashlib
import json
import logging
import os
import re
import urllib.parse as urlparse
import uuid
from base64 import urlsafe_b64encode
from functools import partial
import tornado
from bokeh.server.auth_provider import AuthProvider
from tornado.auth import OAuth2Mixin
from tornado.httpclient import HTTPError as HTTPClientError, HTTPRequest
from tornado.web import HTTPError, RequestHandler, decode_signed_value
from tornado.websocket import WebSocketHandler
from .config import config
from .entry_points import entry_points_for
from .io.resources import (
from .io.state import state
from .util import base64url_encode, decode_token
class OAuthLoginHandler(tornado.web.RequestHandler, OAuth2Mixin):
    _API_BASE_HEADERS = {'Accept': 'application/json', 'User-Agent': 'Tornado OAuth'}
    _DEFAULT_SCOPES = ['openid', 'email', 'profile', 'offline_access']
    _EXTRA_TOKEN_PARAMS = {'grant_type': 'authorization_code'}
    _access_token_header = None
    _state_cookie = None
    _error_template = ERROR_TEMPLATE
    _login_endpoint = '/login'

    @property
    def _SCOPE(self):
        if 'scope' in config.oauth_extra_params:
            return config.oauth_extra_params['scope']
        elif 'PANEL_OAUTH_SCOPE' not in os.environ:
            return self._DEFAULT_SCOPES
        return [scope for scope in os.environ['PANEL_OAUTH_SCOPE'].split(',')]

    async def get_authenticated_user(self, redirect_uri, client_id, state, client_secret=None, code=None):
        """
        Fetches the authenticated user

        Arguments
        ---------
        redirect_uri: (str)
          The OAuth redirect URI
        client_id: (str)
          The OAuth client ID
        state: (str)
          The unguessable random string to protect against
          cross-site request forgery attacks
        client_secret: (str, optional)
          The client secret
        code: (str, optional)
          The response code from the server
        """
        if code:
            user, _, _, _ = await self._fetch_access_token(client_id, redirect_uri, client_secret=client_secret, code=code)
            return user
        params = {'redirect_uri': redirect_uri, 'client_id': client_id, 'client_secret': client_secret, 'response_type': 'code', 'extra_params': {'state': state}}
        if 'audience' in config.oauth_extra_params:
            params['extra_params']['audience'] = config.oauth_extra_params['audience']
        if self._SCOPE is not None:
            params['scope'] = self._SCOPE
        if 'scope' in config.oauth_extra_params:
            params['scope'] = config.oauth_extra_params['scope']
        log.debug('%s making authorize request', type(self).__name__)
        self.authorize_redirect(**params)

    async def _fetch_access_token(self, client_id, redirect_uri=None, client_secret=None, code=None, refresh_token=None, username=None, password=None):
        """
        Fetches the access token.

        Arguments
        ---------
        client_id:
          The client ID
        redirect_uri:
          The redirect URI
        code:
          The response code from the server
        client_secret:
          The client secret
        refresh_token:
          A token used for refreshing the access_token
        username:
          A username
        password:
          A password
        """
        log.debug('%s making access token request.', type(self).__name__)
        params = {'client_id': client_id, **self._EXTRA_TOKEN_PARAMS}
        if redirect_uri:
            params['redirect_uri'] = redirect_uri
        if self._SCOPE:
            params['scope'] = ' '.join(self._SCOPE)
        if code:
            params['code'] = code
        if refresh_token:
            refreshing = True
            params['refresh_token'] = refresh_token
            params['grant_type'] = 'refresh_token'
        else:
            refreshing = False
        if client_secret:
            params['client_secret'] = client_secret
        elif username:
            params.update(username=username, password=password)
        else:
            params['code_verifier'] = self.get_code_cookie()
        http = self.get_auth_http_client()
        req = HTTPRequest(self._OAUTH_ACCESS_TOKEN_URL, method='POST', body=urlparse.urlencode(params), headers=self._API_BASE_HEADERS)
        try:
            response = await http.fetch(req)
        except HTTPClientError as e:
            log.debug('%s access token request failed.', type(self).__name__)
            self._raise_error(e.response, status=401)
        if not response.body or not (body := decode_response_body(response)):
            log.debug('%s token endpoint did not return a valid access token.', type(self).__name__)
            self._raise_error(response)
        if 'access_token' not in body:
            if refresh_token:
                log.debug('%s token endpoint did not reissue an access token.', type(self).__name__)
                return (None, None, None)
            self._raise_error(response, body, status=401)
        expires_in = body.get('expires_in')
        if expires_in:
            expires_in = int(expires_in)
        access_token, refresh_token = (body['access_token'], body.get('refresh_token'))
        if refreshing:
            return (None, access_token, refresh_token, expires_in)
        elif (id_token := body.get('id_token')):
            try:
                user = self._on_auth(id_token, access_token, refresh_token, expires_in)
            except HTTPError:
                pass
            else:
                log.debug('%s successfully obtained access_token and id_token.', type(self).__name__)
                return (user, access_token, refresh_token, expires_in)
        user_headers = dict(self._API_BASE_HEADERS)
        if self._OAUTH_USER_URL:
            if self._access_token_header:
                user_url = self._OAUTH_USER_URL
                user_headers['Authorization'] = self._access_token_header.format(body['access_token'])
            else:
                user_url = '{}{}'.format(self._OAUTH_USER_URL, body['access_token'])
            log.debug('%s requesting OpenID userinfo.', type(self).__name__)
            try:
                user_response = await http.fetch(user_url, headers=user_headers)
                id_token = decode_response_body(user_response)
            except HTTPClientError:
                id_token = None
        if not id_token:
            log.debug('%s could not fetch user information, the token endpoint did not return an id_token and no OpenID user info endpoint was provided. Attempting to code access_token to resolve user information.', type(self).__name__)
            try:
                id_token = decode_token(body['access_token'])
            except Exception:
                log.debug('%s could not decode access_token.', type(self).__name__)
                self._raise_error(response, body, status=401)
        log.debug('%s successfully obtained access_token and userinfo.', type(self).__name__)
        user = self._on_auth(id_token, access_token, refresh_token, expires_in)
        return (user, access_token, refresh_token, expires_in)

    def get_state_cookie(self):
        """Get OAuth state from cookies
        To be compared with the value in redirect URL
        """
        if self._state_cookie is None:
            self._state_cookie = (self.get_secure_cookie(STATE_COOKIE_NAME, max_age_days=config.oauth_expiry) or b'').decode('utf8', 'replace')
            self.clear_cookie(STATE_COOKIE_NAME)
        return self._state_cookie

    def set_state_cookie(self, state):
        self.set_secure_cookie(STATE_COOKIE_NAME, state, expires_days=config.oauth_expiry, httponly=True)

    def get_state(self):
        root_url = self.request.uri.replace(self._login_endpoint, '')
        next_url = original_next_url = self.get_argument('next', root_url)
        if next_url:
            next_url = next_url.replace('\\', urlparse.quote('\\'))
            urlinfo = urlparse.urlparse(next_url)
            next_url = urlinfo._replace(scheme='', netloc='', path='/' + urlinfo.path.lstrip('/')).geturl()
            if next_url != original_next_url:
                log.warning('Ignoring next_url %r, using %r', original_next_url, next_url)
        return _serialize_state({'state_id': uuid.uuid4().hex, 'next_url': next_url or '/'})

    def get_code(self):
        code_verifier = uuid.uuid4().hex + uuid.uuid4().hex + uuid.uuid4().hex
        hashed_code_verifier = hashlib.sha256(code_verifier.encode('utf-8')).digest()
        code_challenge = urlsafe_b64encode(hashed_code_verifier).decode('utf-8').replace('=', '')
        return (code_verifier, code_challenge)

    def get_code_cookie(self):
        code = (self.get_secure_cookie(CODE_COOKIE_NAME, max_age_days=config.oauth_expiry) or b'').decode('utf8', 'replace')
        self.clear_cookie(CODE_COOKIE_NAME)
        return code

    def set_code_cookie(self, code):
        self.set_secure_cookie(CODE_COOKIE_NAME, code, expires_days=config.oauth_expiry, httponly=True)

    async def get(self):
        log.debug('%s received login request', type(self).__name__)
        if config.oauth_redirect_uri:
            redirect_uri = config.oauth_redirect_uri
        else:
            redirect_uri = '{0}://{1}'.format(self.request.protocol, self.request.host)
        params = {'redirect_uri': redirect_uri, 'client_id': config.oauth_key}
        next_arg = self.get_argument('next', {})
        if next_arg:
            next_arg = urlparse.parse_qs(next_arg)
            next_arg = {arg.split('?')[-1]: value for arg, value in next_arg.items()}
        code = self.get_argument('code', extract_urlparam(next_arg, 'code'))
        url_state = self.get_argument('state', extract_urlparam(next_arg, 'state'))
        error = self.get_argument('error', extract_urlparam(next_arg, 'error'))
        if error is not None:
            error_msg = self.get_argument('error_description', extract_urlparam(next_arg, 'error_description'))
            if not error_msg:
                error_msg = error
            log.error('%s failed to authenticate with following error: %s', type(self).__name__, error)
            raise HTTPError(401, error_msg, reason=error)
        cookie_state = self.get_state_cookie()
        if code:
            if cookie_state != url_state:
                log.warning('OAuth state mismatch: %s != %s', cookie_state, url_state)
                raise HTTPError(401, 'OAuth state mismatch. Please restart the authentication flow.', reason='state mismatch')
            state = _deserialize_state(url_state)
            params.update({'client_secret': config.oauth_secret, 'code': code, 'state': url_state})
            user = await self.get_authenticated_user(**params)
            if user is None:
                raise HTTPError(403, 'Permissions unknown.')
            log.debug('%s authorized user, redirecting to app.', type(self).__name__)
            self.redirect(state.get('next_url', '/'))
        else:
            params['state'] = state = self.get_state()
            self.set_state_cookie(state)
            await self.get_authenticated_user(**params)

    def _on_auth(self, id_token, access_token, refresh_token=None, expires_in=None):
        if isinstance(id_token, str):
            decoded = decode_token(id_token)
        else:
            decoded = id_token
            id_token = base64url_encode(json.dumps(id_token))
        user_key = config.oauth_jwt_user or self._USER_KEY
        if user_key in decoded:
            user = decoded[user_key]
        else:
            log.error('%s token payload did not contain expected %r.', type(self).__name__, user_key)
            raise HTTPError(401, 'OAuth token payload missing user information')
        self.clear_cookie('is_guest')
        self.set_secure_cookie('user', user, expires_days=config.oauth_expiry)
        if state.encryption:
            access_token = state.encryption.encrypt(access_token.encode('utf-8'))
            id_token = state.encryption.encrypt(id_token.encode('utf-8'))
            if refresh_token:
                refresh_token = state.encryption.encrypt(refresh_token.encode('utf-8'))
        self.set_secure_cookie('access_token', access_token, expires_days=config.oauth_expiry)
        self.set_secure_cookie('id_token', id_token, expires_days=config.oauth_expiry)
        if expires_in:
            now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
            self.set_secure_cookie('oauth_expiry', str(int(now_ts + expires_in)), expires_days=config.oauth_expiry)
        if refresh_token:
            self.set_secure_cookie('refresh_token', refresh_token, expires_days=config.oauth_expiry)
        if user in state._oauth_user_overrides:
            state._oauth_user_overrides.pop(user, None)
        return user

    def _raise_error(self, response, body=None, status=400):
        try:
            body = body or decode_response_body(response)
        except json.decoder.JSONDecodeError:
            body = body
        provider = self.__class__.__name__.replace('LoginHandler', '')
        if response.error:
            log.error(f'{provider} OAuth provider returned a {response.error} error. The full response was: {body}')
        else:
            log.warning(f'{provider} OAuth provider failed to fully authenticate returning the following response:{body}.')
        raise HTTPError(status, body.get('error_description', str(body)), reason=body.get('error', 'Unknown error'))

    def write_error(self, status_code, **kwargs):
        _, e, _ = kwargs['exc_info']
        self.clear_all_cookies()
        self.set_header('Content-Type', 'text/html')
        if isinstance(e, HTTPError):
            error, error_msg = (e.reason, e.log_message)
        else:
            provider = self.__class__.__name__.replace('LoginHandler', '')
            log.error(f'{provider} OAuth provider encountered unexpected error: {e}')
            error, error_msg = ('500: Internal Server Error', 'Server encountered unexpected problem.')
        self.write(self._error_template.render(npm_cdn=config.npm_cdn, title='Panel: Authentication Error', error_type='Authentication Error', error=error, error_msg=error_msg))