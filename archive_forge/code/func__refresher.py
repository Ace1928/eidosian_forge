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