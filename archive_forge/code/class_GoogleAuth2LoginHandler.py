import json
import os
import re
import uuid
from urllib.parse import urlencode
import tornado.auth
import tornado.gen
import tornado.web
from celery.utils.imports import instantiate
from tornado.options import options
from ..views import BaseHandler
from ..views.error import NotFoundErrorHandler
class GoogleAuth2LoginHandler(BaseHandler, tornado.auth.GoogleOAuth2Mixin):
    _OAUTH_SETTINGS_KEY = 'oauth'

    async def get(self):
        redirect_uri = self.settings[self._OAUTH_SETTINGS_KEY]['redirect_uri']
        if self.get_argument('code', False):
            user = await self.get_authenticated_user(redirect_uri=redirect_uri, code=self.get_argument('code'))
            await self._on_auth(user)
        else:
            self.authorize_redirect(redirect_uri=redirect_uri, client_id=self.settings[self._OAUTH_SETTINGS_KEY]['key'], scope=['profile', 'email'], response_type='code', extra_params={'approval_prompt': ''})

    async def _on_auth(self, user):
        if not user:
            raise tornado.web.HTTPError(403, 'Google auth failed')
        access_token = user['access_token']
        try:
            response = await self.get_auth_http_client().fetch('https://www.googleapis.com/userinfo/v2/me', headers={'Authorization': f'Bearer {access_token}'})
        except Exception as e:
            raise tornado.web.HTTPError(403, f'Google auth failed: {e}')
        email = json.loads(response.body.decode('utf-8'))['email']
        if not authenticate(self.application.options.auth, email):
            message = f"Access denied to '{email}'. Please use another account or ask your admin to add your email to flower --auth."
            raise tornado.web.HTTPError(403, message)
        self.set_secure_cookie('user', str(email))
        next_ = self.get_argument('next', self.application.options.url_prefix or '/')
        if self.application.options.url_prefix and next_[0] != '/':
            next_ = '/' + next_
        self.redirect(next_)