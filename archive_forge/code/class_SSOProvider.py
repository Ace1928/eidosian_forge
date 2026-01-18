import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
class SSOProvider(CredentialProvider):
    METHOD = 'sso'
    _SSO_TOKEN_CACHE_DIR = os.path.expanduser(os.path.join('~', '.aws', 'sso', 'cache'))
    _PROFILE_REQUIRED_CONFIG_VARS = ('sso_role_name', 'sso_account_id')
    _SSO_REQUIRED_CONFIG_VARS = ('sso_start_url', 'sso_region')
    _ALL_REQUIRED_CONFIG_VARS = _PROFILE_REQUIRED_CONFIG_VARS + _SSO_REQUIRED_CONFIG_VARS

    def __init__(self, load_config, client_creator, profile_name, cache=None, token_cache=None, token_provider=None):
        if token_cache is None:
            token_cache = JSONFileCache(self._SSO_TOKEN_CACHE_DIR)
        self._token_cache = token_cache
        self._token_provider = token_provider
        if cache is None:
            cache = {}
        self.cache = cache
        self._load_config = load_config
        self._client_creator = client_creator
        self._profile_name = profile_name

    def _load_sso_config(self):
        loaded_config = self._load_config()
        profiles = loaded_config.get('profiles', {})
        profile_name = self._profile_name
        profile_config = profiles.get(self._profile_name, {})
        sso_sessions = loaded_config.get('sso_sessions', {})
        if all((c not in profile_config for c in self._PROFILE_REQUIRED_CONFIG_VARS)):
            return None
        resolved_config, extra_reqs = self._resolve_sso_session_reference(profile_config, sso_sessions)
        config = {}
        missing_config_vars = []
        all_required_configs = self._ALL_REQUIRED_CONFIG_VARS + extra_reqs
        for config_var in all_required_configs:
            if config_var in resolved_config:
                config[config_var] = resolved_config[config_var]
            else:
                missing_config_vars.append(config_var)
        if missing_config_vars:
            missing = ', '.join(missing_config_vars)
            raise InvalidConfigError(error_msg='The profile "%s" is configured to use SSO but is missing required configuration: %s' % (profile_name, missing))
        return config

    def _resolve_sso_session_reference(self, profile_config, sso_sessions):
        sso_session_name = profile_config.get('sso_session')
        if sso_session_name is None:
            return (profile_config, ())
        if sso_session_name not in sso_sessions:
            error_msg = f'The specified sso-session does not exist: "{sso_session_name}"'
            raise InvalidConfigError(error_msg=error_msg)
        config = profile_config.copy()
        session = sso_sessions[sso_session_name]
        for config_var, val in session.items():
            if config.get(config_var, val) != val:
                error_msg = f'The value for {config_var} is inconsistent between profile ({config[config_var]}) and sso-session ({val}).'
                raise InvalidConfigError(error_msg=error_msg)
            config[config_var] = val
        return (config, ('sso_session',))

    def load(self):
        sso_config = self._load_sso_config()
        if not sso_config:
            return None
        fetcher_kwargs = {'start_url': sso_config['sso_start_url'], 'sso_region': sso_config['sso_region'], 'role_name': sso_config['sso_role_name'], 'account_id': sso_config['sso_account_id'], 'client_creator': self._client_creator, 'token_loader': SSOTokenLoader(cache=self._token_cache), 'cache': self.cache}
        if 'sso_session' in sso_config:
            fetcher_kwargs['sso_session_name'] = sso_config['sso_session']
            fetcher_kwargs['token_provider'] = self._token_provider
        sso_fetcher = SSOCredentialFetcher(**fetcher_kwargs)
        return DeferredRefreshableCredentials(method=self.METHOD, refresh_using=sso_fetcher.fetch_credentials)