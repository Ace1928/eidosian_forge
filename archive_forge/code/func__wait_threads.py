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
def _wait_threads(self):
    """
        Tell all the threads to terminate (by sending a sentinel value) and
        wait for them to do so.
        """
    for t in self._threads:
        self._to_fetch.put(None)
    for t in self._threads:
        t.join()
    self._threads = []