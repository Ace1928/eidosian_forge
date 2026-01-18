import enum
import functools
import itertools
import logging
import re
from typing import TYPE_CHECKING, FrozenSet, Iterable, List, Optional, Set, Tuple, Union
from pip._vendor.packaging import specifiers
from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import _BaseVersion
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.exceptions import (
from pip._internal.index.collector import LinkCollector, parse_links
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.format_control import FormatControl
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.models.wheel import Wheel
from pip._internal.req import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import build_netloc
from pip._internal.utils.packaging import check_requires_python
from pip._internal.utils.unpacking import SUPPORTED_EXTENSIONS
def _check_link_requires_python(link: Link, version_info: Tuple[int, int, int], ignore_requires_python: bool=False) -> bool:
    """
    Return whether the given Python version is compatible with a link's
    "Requires-Python" value.

    :param version_info: A 3-tuple of ints representing the Python
        major-minor-micro version to check.
    :param ignore_requires_python: Whether to ignore the "Requires-Python"
        value if the given Python version isn't compatible.
    """
    try:
        is_compatible = check_requires_python(link.requires_python, version_info=version_info)
    except specifiers.InvalidSpecifier:
        logger.debug('Ignoring invalid Requires-Python (%r) for link: %s', link.requires_python, link)
    else:
        if not is_compatible:
            version = '.'.join(map(str, version_info))
            if not ignore_requires_python:
                logger.verbose('Link requires a different Python (%s not in: %r): %s', version, link.requires_python, link)
                return False
            logger.debug('Ignoring failed Requires-Python check (%s not in: %r) for link: %s', version, link.requires_python, link)
    return True