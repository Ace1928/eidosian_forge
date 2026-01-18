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
class KeyRingPythonProvider(KeyRingBaseProvider):
    """Keyring interface which uses locally imported `keyring`"""
    has_keyring = True

    def __init__(self) -> None:
        import keyring
        self.keyring = keyring

    def get_auth_info(self, url: str, username: Optional[str]) -> Optional[AuthInfo]:
        if hasattr(self.keyring, 'get_credential'):
            logger.debug('Getting credentials from keyring for %s', url)
            cred = self.keyring.get_credential(url, username)
            if cred is not None:
                return (cred.username, cred.password)
            return None
        if username is not None:
            logger.debug('Getting password from keyring for %s', url)
            password = self.keyring.get_password(url, username)
            if password:
                return (username, password)
        return None

    def save_auth_info(self, url: str, username: str, password: str) -> None:
        self.keyring.set_password(url, username, password)