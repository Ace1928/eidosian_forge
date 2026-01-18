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
def _prepare_threads(self):
    """
        Threads are created only when get_project is called, and terminate
        before it returns. They are there primarily to parallelise I/O (i.e.
        fetching web pages).
        """
    self._threads = []
    for i in range(self.num_workers):
        t = threading.Thread(target=self._fetch)
        t.daemon = True
        t.start()
        self._threads.append(t)