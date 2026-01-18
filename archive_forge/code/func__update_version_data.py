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
def _update_version_data(self, result, info):
    """
        Update a result dictionary (the final result from _get_project) with a
        dictionary for a specific version, which typically holds information
        gleaned from a filename or URL for an archive for the distribution.
        """
    name = info.pop('name')
    version = info.pop('version')
    if version in result:
        dist = result[version]
        md = dist.metadata
    else:
        dist = make_dist(name, version, scheme=self.scheme)
        md = dist.metadata
    dist.digest = digest = self._get_digest(info)
    url = info['url']
    result['digests'][url] = digest
    if md.source_url != info['url']:
        md.source_url = self.prefer_url(md.source_url, url)
        result['urls'].setdefault(version, set()).add(url)
    dist.locator = self
    result[version] = dist