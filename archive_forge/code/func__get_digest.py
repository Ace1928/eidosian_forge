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
def _get_digest(self, info):
    """
        Get a digest from a dictionary by looking at a "digests" dictionary
        or keys of the form 'algo_digest'.

        Returns a 2-tuple (algo, digest) if found, else None. Currently
        looks only for SHA256, then MD5.
        """
    result = None
    if 'digests' in info:
        digests = info['digests']
        for algo in ('sha256', 'md5'):
            if algo in digests:
                result = (algo, digests[algo])
                break
    if not result:
        for algo in ('sha256', 'md5'):
            key = '%s_digest' % algo
            if key in info:
                result = (algo, info[key])
                break
    return result