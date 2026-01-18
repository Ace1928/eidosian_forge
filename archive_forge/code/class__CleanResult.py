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
class _CleanResult(NamedTuple):
    """Convert link for equivalency check.

    This is used in the resolver to check whether two URL-specified requirements
    likely point to the same distribution and can be considered equivalent. This
    equivalency logic avoids comparing URLs literally, which can be too strict
    (e.g. "a=1&b=2" vs "b=2&a=1") and produce conflicts unexpecting to users.

    Currently this does three things:

    1. Drop the basic auth part. This is technically wrong since a server can
       serve different content based on auth, but if it does that, it is even
       impossible to guarantee two URLs without auth are equivalent, since
       the user can input different auth information when prompted. So the
       practical solution is to assume the auth doesn't affect the response.
    2. Parse the query to avoid the ordering issue. Note that ordering under the
       same key in the query are NOT cleaned; i.e. "a=1&a=2" and "a=2&a=1" are
       still considered different.
    3. Explicitly drop most of the fragment part, except ``subdirectory=`` and
       hash values, since it should have no impact the downloaded content. Note
       that this drops the "egg=" part historically used to denote the requested
       project (and extras), which is wrong in the strictest sense, but too many
       people are supplying it inconsistently to cause superfluous resolution
       conflicts, so we choose to also ignore them.
    """
    parsed: urllib.parse.SplitResult
    query: Dict[str, List[str]]
    subdirectory: str
    hashes: Dict[str, str]