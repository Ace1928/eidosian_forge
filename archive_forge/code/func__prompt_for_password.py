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
def _prompt_for_password(self, netloc: str) -> Tuple[Optional[str], Optional[str], bool]:
    username = ask_input(f'User for {netloc}: ') if self.prompting else None
    if not username:
        return (None, None, False)
    if self.use_keyring:
        auth = self._get_keyring_auth(netloc, username)
        if auth and auth[0] is not None and (auth[1] is not None):
            return (auth[0], auth[1], False)
    password = ask_password('Password: ')
    return (username, password, True)