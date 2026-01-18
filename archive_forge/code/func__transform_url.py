import contextlib
import errno
import getpass
import hashlib
import io
import logging
import os
import posixpath
import shutil
import stat
import sys
import sysconfig
import urllib.parse
from functools import partial
from io import StringIO
from itertools import filterfalse, tee, zip_longest
from pathlib import Path
from types import FunctionType, TracebackType
from typing import (
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.pyproject_hooks import BuildBackendHookCaller
from pip._vendor.tenacity import retry, stop_after_delay, wait_fixed
from pip import __version__
from pip._internal.exceptions import CommandError, ExternallyManagedEnvironment
from pip._internal.locations import get_major_minor_version
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.virtualenv import running_under_virtualenv
def _transform_url(url: str, transform_netloc: Callable[[str], Tuple[Any, ...]]) -> Tuple[str, NetlocTuple]:
    """Transform and replace netloc in a url.

    transform_netloc is a function taking the netloc and returning a
    tuple. The first element of this tuple is the new netloc. The
    entire tuple is returned.

    Returns a tuple containing the transformed url as item 0 and the
    original tuple returned by transform_netloc as item 1.
    """
    purl = urllib.parse.urlsplit(url)
    netloc_tuple = transform_netloc(purl.netloc)
    url_pieces = (purl.scheme, netloc_tuple[0], purl.path, purl.query, purl.fragment)
    surl = urllib.parse.urlunsplit(url_pieces)
    return (surl, cast('NetlocTuple', netloc_tuple))