import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
def _cache_key(self, key, options, pos, size, parent_keys):
    """Cache a version record in the history array and index cache.

        This is inlined into _load_data for performance. KEEP IN SYNC.
        (It saves 60ms, 25% of the __init__ overhead on local 4000 record
         indexes).
        """
    prefix = key[:-1]
    version_id = key[-1]
    parents = tuple((parent[-1] for parent in parent_keys))
    for parent in parent_keys:
        if parent[:-1] != prefix:
            raise ValueError('mismatched prefixes for {!r}, {!r}'.format(key, parent_keys))
    cache, history = self._kndx_cache[prefix]
    if version_id not in cache:
        index = len(history)
        history.append(version_id)
    else:
        index = cache[version_id][5]
    cache[version_id] = (version_id, options, pos, size, parents, index)