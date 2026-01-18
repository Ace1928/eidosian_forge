from __future__ import unicode_literals
import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport
from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME,
from .util import (parse_requirement, cached_property, parse_name_and_version,
def _generate_cache(self):
    """
        Scan the path for distributions and populate the cache with
        those that are found.
        """
    gen_dist = not self._cache.generated
    gen_egg = self._include_egg and (not self._cache_egg.generated)
    if gen_dist or gen_egg:
        for dist in self._yield_distributions():
            if isinstance(dist, InstalledDistribution):
                self._cache.add(dist)
            else:
                self._cache_egg.add(dist)
        if gen_dist:
            self._cache.generated = True
        if gen_egg:
            self._cache_egg.generated = True