import logging
import os
import shutil
import subprocess
import sysconfig
import typing
import urllib.parse
from abc import ABC, abstractmethod
from functools import lru_cache
from os.path import commonprefix
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from pip._vendor.requests.auth import AuthBase, HTTPBasicAuth
from pip._vendor.requests.models import Request, Response
from pip._vendor.requests.utils import get_netrc_auth
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
from pip._internal.vcs.versioncontrol import AuthInfo
def _get_new_credentials(self, original_url: str, *, allow_netrc: bool=True, allow_keyring: bool=False) -> AuthInfo:
    """Find and return credentials for the specified URL."""
    url, netloc, url_user_password = split_auth_netloc_from_url(original_url)
    username, password = url_user_password
    if username is not None and password is not None:
        logger.debug('Found credentials in url for %s', netloc)
        return url_user_password
    index_url = self._get_index_url(url)
    if index_url:
        index_info = split_auth_netloc_from_url(index_url)
        if index_info:
            index_url, _, index_url_user_password = index_info
            logger.debug('Found index url %s', index_url)
    if index_url and index_url_user_password[0] is not None:
        username, password = index_url_user_password
        if username is not None and password is not None:
            logger.debug('Found credentials in index url for %s', netloc)
            return index_url_user_password
    if allow_netrc:
        netrc_auth = get_netrc_auth(original_url)
        if netrc_auth:
            logger.debug('Found credentials in netrc for %s', netloc)
            return netrc_auth
    if allow_keyring:
        kr_auth = self._get_keyring_auth(index_url, username) or self._get_keyring_auth(netloc, username)
        if kr_auth:
            logger.debug('Found credentials in keyring for %s', netloc)
            return kr_auth
    return (username, password)