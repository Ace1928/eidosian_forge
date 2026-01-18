import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import NamedTuple, Optional
import dateutil.parser
from dateutil.tz import tzutc
from botocore import UNSIGNED
from botocore.compat import total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.utils import CachedProperty, JSONFileCache, SSOTokenLoader
class SSOTokenProvider:
    METHOD = 'sso'
    _REFRESH_WINDOW = 15 * 60
    _SSO_TOKEN_CACHE_DIR = os.path.expanduser(os.path.join('~', '.aws', 'sso', 'cache'))
    _SSO_CONFIG_VARS = ['sso_start_url', 'sso_region']
    _GRANT_TYPE = 'refresh_token'
    DEFAULT_CACHE_CLS = JSONFileCache

    def __init__(self, session, cache=None, time_fetcher=_utc_now, profile_name=None):
        self._session = session
        if cache is None:
            cache = self.DEFAULT_CACHE_CLS(self._SSO_TOKEN_CACHE_DIR, dumps_func=_sso_json_dumps)
        self._now = time_fetcher
        self._cache = cache
        self._token_loader = SSOTokenLoader(cache=self._cache)
        self._profile_name = profile_name or self._session.get_config_variable('profile') or 'default'

    def _load_sso_config(self):
        loaded_config = self._session.full_config
        profiles = loaded_config.get('profiles', {})
        sso_sessions = loaded_config.get('sso_sessions', {})
        profile_config = profiles.get(self._profile_name, {})
        if 'sso_session' not in profile_config:
            return
        sso_session_name = profile_config['sso_session']
        sso_config = sso_sessions.get(sso_session_name, None)
        if not sso_config:
            error_msg = f'The profile "{self._profile_name}" is configured to use the SSO token provider but the "{sso_session_name}" sso_session configuration does not exist.'
            raise InvalidConfigError(error_msg=error_msg)
        missing_configs = []
        for var in self._SSO_CONFIG_VARS:
            if var not in sso_config:
                missing_configs.append(var)
        if missing_configs:
            error_msg = f'The profile "{self._profile_name}" is configured to use the SSO token provider but is missing the following configuration: {missing_configs}.'
            raise InvalidConfigError(error_msg=error_msg)
        return {'session_name': sso_session_name, 'sso_region': sso_config['sso_region'], 'sso_start_url': sso_config['sso_start_url']}

    @CachedProperty
    def _sso_config(self):
        return self._load_sso_config()

    @CachedProperty
    def _client(self):
        config = Config(region_name=self._sso_config['sso_region'], signature_version=UNSIGNED)
        return self._session.create_client('sso-oidc', config=config)

    def _attempt_create_token(self, token):
        response = self._client.create_token(grantType=self._GRANT_TYPE, clientId=token['clientId'], clientSecret=token['clientSecret'], refreshToken=token['refreshToken'])
        expires_in = timedelta(seconds=response['expiresIn'])
        new_token = {'startUrl': self._sso_config['sso_start_url'], 'region': self._sso_config['sso_region'], 'accessToken': response['accessToken'], 'expiresAt': self._now() + expires_in, 'clientId': token['clientId'], 'clientSecret': token['clientSecret'], 'registrationExpiresAt': token['registrationExpiresAt']}
        if 'refreshToken' in response:
            new_token['refreshToken'] = response['refreshToken']
        logger.info('SSO Token refresh succeeded')
        return new_token

    def _refresh_access_token(self, token):
        keys = ('refreshToken', 'clientId', 'clientSecret', 'registrationExpiresAt')
        missing_keys = [k for k in keys if k not in token]
        if missing_keys:
            msg = f'Unable to refresh SSO token: missing keys: {missing_keys}'
            logger.info(msg)
            return None
        expiry = dateutil.parser.parse(token['registrationExpiresAt'])
        if total_seconds(expiry - self._now()) <= 0:
            logger.info(f'SSO token registration expired at {expiry}')
            return None
        try:
            return self._attempt_create_token(token)
        except ClientError:
            logger.warning('SSO token refresh attempt failed', exc_info=True)
            return None

    def _refresher(self):
        start_url = self._sso_config['sso_start_url']
        session_name = self._sso_config['session_name']
        logger.info(f'Loading cached SSO token for {session_name}')
        token_dict = self._token_loader(start_url, session_name=session_name)
        expiration = dateutil.parser.parse(token_dict['expiresAt'])
        logger.debug(f'Cached SSO token expires at {expiration}')
        remaining = total_seconds(expiration - self._now())
        if remaining < self._REFRESH_WINDOW:
            new_token_dict = self._refresh_access_token(token_dict)
            if new_token_dict is not None:
                token_dict = new_token_dict
                expiration = token_dict['expiresAt']
                self._token_loader.save_token(start_url, token_dict, session_name=session_name)
        return FrozenAuthToken(token_dict['accessToken'], expiration=expiration)

    def load_token(self):
        if self._sso_config is None:
            return None
        return DeferredRefreshableToken(self.METHOD, self._refresher, time_fetcher=self._now)