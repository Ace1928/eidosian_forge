from __future__ import absolute_import, division, print_function, unicode_literals
import codecs
from collections import defaultdict
from math import ceil, log as logf
import logging; log = logging.getLogger(__name__)
import pkg_resources
import os
from passlib import exc
from passlib.utils.compat import PY2, irange, itervalues, int_types
from passlib.utils import rng, getrandstr, to_unicode
from passlib.utils.decor import memoized_property
def _open_asset_path(path, encoding=None):
    """
    :param asset_path:
        string containing absolute path to file,
        or package-relative path using format
        ``"python.module:relative/file/path"``.

    :returns:
        filehandle opened in 'rb' mode
        (unless encoding explicitly specified)
    """
    if encoding:
        return codecs.getreader(encoding)(_open_asset_path(path))
    if os.path.isabs(path):
        return open(path, 'rb')
    package, sep, subpath = path.partition(':')
    if not sep:
        raise ValueError("asset path must be absolute file path or use 'pkg.name:sub/path' format: %r" % (path,))
    return pkg_resources.resource_stream(package, subpath)