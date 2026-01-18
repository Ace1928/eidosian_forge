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
def _clean_link(link: Link) -> _CleanResult:
    parsed = link._parsed_url
    netloc = parsed.netloc.rsplit('@', 1)[-1]
    if parsed.scheme == 'file' and (not netloc):
        netloc = 'localhost'
    fragment = urllib.parse.parse_qs(parsed.fragment)
    if 'egg' in fragment:
        logger.debug('Ignoring egg= fragment in %s', link)
    try:
        subdirectory = fragment['subdirectory'][0]
    except (IndexError, KeyError):
        subdirectory = ''
    hashes = {k: fragment[k][0] for k in _SUPPORTED_HASHES if k in fragment}
    return _CleanResult(parsed=parsed._replace(netloc=netloc, query='', fragment=''), query=urllib.parse.parse_qs(parsed.query), subdirectory=subdirectory, hashes=hashes)