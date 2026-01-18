import gzip
from io import BytesIO
import json
import logging
import os
import posixpath
import re
import zlib
from . import DistlibException
from .compat import (urljoin, urlparse, urlunparse, url2pathname, pathname2url,
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import (cached_property, ensure_slash, split_filename, get_project_data,
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible
def add_distribution(self, dist):
    """
        Add a distribution to the finder. This will update internal information
        about who provides what.
        :param dist: The distribution to add.
        """
    logger.debug('adding distribution %s', dist)
    name = dist.key
    self.dists_by_name[name] = dist
    self.dists[name, dist.version] = dist
    for p in dist.provides:
        name, version = parse_name_and_version(p)
        logger.debug('Add to provided: %s, %s, %s', name, version, dist)
        self.provided.setdefault(name, set()).add((version, dist))