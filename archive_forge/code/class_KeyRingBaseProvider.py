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
class KeyRingBaseProvider(ABC):
    """Keyring base provider interface"""
    has_keyring: bool

    @abstractmethod
    def get_auth_info(self, url: str, username: Optional[str]) -> Optional[AuthInfo]:
        ...

    @abstractmethod
    def save_auth_info(self, url: str, username: str, password: str) -> None:
        ...