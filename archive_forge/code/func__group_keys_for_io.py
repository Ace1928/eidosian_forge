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
def _group_keys_for_io(self, keys, non_local_keys, positions, _min_buffer_size=_STREAM_MIN_BUFFER_SIZE):
    """For the given keys, group them into 'best-sized' requests.

        The idea is to avoid making 1 request per file, but to never try to
        unpack an entire 1.5GB source tree in a single pass. Also when
        possible, we should try to group requests to the same pack file
        together.

        :return: list of (keys, non_local) tuples that indicate what keys
            should be fetched next.
        """
    total_keys = len(keys)
    prefix_split_keys, prefix_order = self._split_by_prefix(keys)
    prefix_split_non_local_keys, _ = self._split_by_prefix(non_local_keys)
    cur_keys = []
    cur_non_local = set()
    cur_size = 0
    result = []
    sizes = []
    for prefix in prefix_order:
        keys = prefix_split_keys[prefix]
        non_local = prefix_split_non_local_keys.get(prefix, [])
        this_size = self._index._get_total_build_size(keys, positions)
        cur_size += this_size
        cur_keys.extend(keys)
        cur_non_local.update(non_local)
        if cur_size > _min_buffer_size:
            result.append((cur_keys, cur_non_local))
            sizes.append(cur_size)
            cur_keys = []
            cur_non_local = set()
            cur_size = 0
    if cur_keys:
        result.append((cur_keys, cur_non_local))
        sizes.append(cur_size)
    return result