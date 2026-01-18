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
class OAuthProvider(BasicAuthProvider):
    """
    An AuthProvider using specific OAuth implementation selected via
    the global config.oauth_provider configuration.
    """

    @property
    def get_user(self):
        return None

    @property
    def get_user_async(self):

        async def get_user(handler):
            user = super(OAuthProvider, self).get_user(handler)
            if not config.oauth_refresh_tokens or user is None:
                return user
            now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
            expiry = None
            if user in state._oauth_user_overrides:
                while not state._oauth_user_overrides[user]:
                    await asyncio.sleep(0.1)
                user_state = state._oauth_user_overrides[user]
                access_token = user_state['access_token']
                if user_state['expiry']:
                    expiry = user_state['expiry']
            else:
                access_cookie = handler.get_secure_cookie('access_token', max_age_days=config.oauth_expiry)
                if not access_cookie:
                    log.debug('No access token available, forcing user to reauthenticate.')
                    return
                access_token = state._decrypt_cookie(access_cookie)
            if expiry is None:
                try:
                    access_json = decode_token(access_token)
                    expiry = access_json['exp']
                except Exception:
                    expiry = handler.get_secure_cookie('oauth_expiry', max_age_days=config.oauth_expiry)
                    if expiry is None:
                        log.debug('access_token is not a valid JWT token. Expiry cannot be determined.')
                        return user
            if user in state._oauth_user_overrides:
                refresh_token = state._oauth_user_overrides[user]['refresh_token']
            else:
                refresh_cookie = handler.get_secure_cookie('refresh_token', max_age_days=config.oauth_expiry)
                if refresh_cookie:
                    refresh_token = state._decrypt_cookie(refresh_cookie)
                    self._schedule_refresh(access_json['exp'], user, refresh_token, handler.application, handler.request)
                else:
                    refresh_token = None
            if expiry > now_ts:
                log.debug('Fully authenticated and access_token still valid.')
                return user
            if refresh_token:
                try:
                    refresh_json = decode_token(refresh_token)
                    if refresh_json['exp'] < now_ts:
                        refresh_token = None
                except Exception:
                    pass
            if refresh_token is None:
                log.debug('%s access_token is expired and refresh_token not available, forcing user to reauthenticate.', type(self).__name__)
                return
            log.debug('%s refreshing token', type(self).__name__)
            await self._refresh_access_token(user, refresh_token, handler.application, handler.request)
            return user
        return get_user

    @property
    def login_handler(self):
        handler = AUTH_PROVIDERS[config.oauth_provider]
        if self._error_template:
            handler._error_template = self._error_template
        handler._login_template = self._login_template
        handler._login_endpoint = self._login_endpoint
        return handler

    def _remove_user(self, session_context):
        guest_cookie = session_context.request.cookies.get('is_guest')
        user_cookie = session_context.request.cookies.get('user')
        if guest_cookie:
            user = 'guest'
        elif user_cookie:
            user = decode_signed_value(config.cookie_secret, 'user', user_cookie)
            if user:
                user = user.decode('utf-8')
        else:
            user = None
        if not user:
            return
        state._active_users[user] -= 1
        if not state._active_users[user]:
            del state._active_users[user]
            if user in state._oauth_user_overrides:
                del state._oauth_user_overrides[user]

    def _schedule_refresh(self, expiry_ts, user, refresh_token, application, request):
        if not state._active_users.get(user):
            return
        now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
        expiry_seconds = expiry_ts - now_ts - 10
        log.debug('%s scheduling token refresh in %d seconds', type(self).__name__, expiry_seconds)
        expiry_date = dt.datetime.now() + dt.timedelta(seconds=expiry_seconds)
        refresh_cb = partial(self._scheduled_refresh, user, refresh_token, application, request)
        if expiry_seconds <= 0:
            state.execute(refresh_cb)
            return
        task = f'{user}-refresh-access-tokens'
        try:
            state.cancel_task(task)
        except KeyError:
            pass
        finally:
            state.schedule_task(task, refresh_cb, at=expiry_date)

    async def _scheduled_refresh(self, user, refresh_token, application, request):
        await self._refresh_access_token(user, refresh_token, application, request)
        user_state = state._oauth_user_overrides[user]
        access_token, refresh_token = (user_state['access_token'], user_state['refresh_token'])
        if user_state['expiry']:
            expiry = user_state['expiry']
        else:
            expiry = decode_token(access_token)['exp']
        self._schedule_refresh(expiry, user, refresh_token, application, request)

    async def _refresh_access_token(self, user, refresh_token, application, request):
        if user in state._oauth_user_overrides:
            if not state._oauth_user_overrides[user]:
                while not state._oauth_user_overrides[user]:
                    await asyncio.sleep(0.1)
                return
            else:
                refresh_token = state._oauth_user_overrides[user]['refresh_token']
        log.debug('%s refreshing token', type(self).__name__)
        state._oauth_user_overrides[user] = {}
        auth_handler = self.login_handler(application=application, request=request)
        _, access_token, refresh_token, expires_in = await auth_handler._fetch_access_token(client_id=config.oauth_key, client_secret=config.oauth_secret, refresh_token=refresh_token)
        if access_token:
            now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
            state._oauth_user_overrides[user] = {'access_token': access_token, 'refresh_token': refresh_token, 'expiry': now_ts + expires_in if expires_in else None}
        else:
            del state._oauth_user_overrides[user]