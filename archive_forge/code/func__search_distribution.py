import email.message
import email.parser
import logging
import os
import zipfile
from typing import Collection, Iterable, Iterator, List, Mapping, NamedTuple, Optional
from pip._vendor import pkg_resources
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.exceptions import InvalidWheel, NoneMetadataError, UnsupportedWheel
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.misc import display_path, normalize_path
from pip._internal.utils.wheel import parse_wheel, read_wheel_metadata_file
from .base import (
def _search_distribution(self, name: str) -> Optional[BaseDistribution]:
    """Find a distribution matching the ``name`` in the environment.

        This searches from *all* distributions available in the environment, to
        match the behavior of ``pkg_resources.get_distribution()``.
        """
    canonical_name = canonicalize_name(name)
    for dist in self.iter_all_distributions():
        if dist.canonical_name == canonical_name:
            return dist
    return None