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
@typing.no_type_check
def PATH_as_shutil_which_determines_it() -> str:
    path = os.environ.get('PATH', None)
    if path is None:
        try:
            path = os.confstr('CS_PATH')
        except (AttributeError, ValueError):
            path = os.defpath
    return path