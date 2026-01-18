import csv
import email.message
import functools
import json
import logging
import pathlib
import re
import zipfile
from typing import (
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import LegacyVersion, Version
from pip._internal.exceptions import NoneMetadataError
from pip._internal.locations import site_packages, user_site
from pip._internal.models.direct_url import (
from pip._internal.utils.compat import stdlib_pkgs  # TODO: Move definition here.
from pip._internal.utils.egg_link import egg_link_path_from_sys_path
from pip._internal.utils.misc import is_local, normalize_path
from pip._internal.utils.urls import url_to_path
from ._json import msg_to_json
def _iter_requires_txt_entries(self) -> Iterator[RequiresEntry]:
    """Parse a ``requires.txt`` in an egg-info directory.

        This is an INI-ish format where an egg-info stores dependencies. A
        section name describes extra other environment markers, while each entry
        is an arbitrary string (not a key-value pair) representing a dependency
        as a requirement string (no markers).

        There is a construct in ``importlib.metadata`` called ``Sectioned`` that
        does mostly the same, but the format is currently considered private.
        """
    try:
        content = self.read_text('requires.txt')
    except FileNotFoundError:
        return
    extra = marker = ''
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            extra, _, marker = line.strip('[]').partition(':')
            continue
        yield RequiresEntry(requirement=line, extra=extra, marker=marker)