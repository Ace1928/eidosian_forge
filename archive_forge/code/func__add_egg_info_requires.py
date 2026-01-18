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
def _add_egg_info_requires(self, metadata: email.message.Message) -> None:
    """Add egg-info requires.txt information to the metadata."""
    if not metadata.get_all('Requires-Dist'):
        for dep in self._iter_egg_info_dependencies():
            metadata['Requires-Dist'] = dep
    if not metadata.get_all('Provides-Extra'):
        for extra in self._iter_egg_info_extras():
            metadata['Provides-Extra'] = extra