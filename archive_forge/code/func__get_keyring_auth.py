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
def _get_keyring_auth(self, url: Optional[str], username: Optional[str]) -> Optional[AuthInfo]:
    """Return the tuple auth for a given url from keyring."""
    if not url:
        return None
    try:
        return self.keyring_provider.get_auth_info(url, username)
    except Exception as exc:
        logger.warning('Keyring is skipped due to an exception: %s', str(exc))
        global KEYRING_DISABLED
        KEYRING_DISABLED = True
        get_keyring_provider.cache_clear()
        return None