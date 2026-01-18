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
def _load_wordset(asset_path):
    """
    load wordset from compressed datafile within package data.
    file should be utf-8 encoded

    :param asset_path:
        string containing  absolute path to wordset file,
        or "python.module:relative/file/path".

    :returns:
        tuple of words, as loaded from specified words file.
    """
    with _open_asset_path(asset_path, 'utf-8') as fh:
        gen = (word.strip() for word in fh)
        words = tuple((word for word in gen if word))
    log.debug('loaded %d-element wordset from %r', len(words), asset_path)
    return words