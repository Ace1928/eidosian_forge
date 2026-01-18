import functools
import itertools
import logging
import os
import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from typing import (
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
from pip._internal.utils.models import KeyBasedCompareMixin
from pip._internal.utils.urls import path_to_url, url_to_path
def _clean_url_path_part(part: str) -> str:
    """
    Clean a "part" of a URL path (i.e. after splitting on "@" characters).
    """
    return urllib.parse.quote(urllib.parse.unquote(part))