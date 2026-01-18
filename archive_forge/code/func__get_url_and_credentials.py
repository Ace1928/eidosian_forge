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
def _get_url_and_credentials(self, original_url: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Return the credentials to use for the provided URL.

        If allowed, netrc and keyring may be used to obtain the
        correct credentials.

        Returns (url_without_credentials, username, password). Note
        that even if the original URL contains credentials, this
        function may return a different username and password.
        """
    url, netloc, _ = split_auth_netloc_from_url(original_url)
    username, password = self._get_new_credentials(original_url)
    if (username is None or password is None) and netloc in self.passwords:
        un, pw = self.passwords[netloc]
        if username is None or username == un:
            username, password = (un, pw)
    if username is not None or password is not None:
        username = username or ''
        password = password or ''
        self.passwords[netloc] = (username, password)
    assert username is not None and password is not None or (username is None and password is None), f'Could not load credentials from url: {original_url}'
    return (url, username, password)