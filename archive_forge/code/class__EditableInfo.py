import collections
import logging
import os
from typing import Container, Dict, Generator, Iterable, List, NamedTuple, Optional, Set
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.req.constructors import (
from pip._internal.req.req_file import COMMENT_RE
from pip._internal.utils.direct_url_helpers import direct_url_as_pep440_direct_reference
class _EditableInfo(NamedTuple):
    requirement: str
    comments: List[str]