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
@classmethod
def _split_by_prefix(cls, keys):
    """For the given keys, split them up based on their prefix.

        To keep memory pressure somewhat under control, split the
        requests back into per-file-id requests, otherwise "bzr co"
        extracts the full tree into memory before writing it to disk.
        This should be revisited if _get_content_maps() can ever cross
        file-id boundaries.

        The keys for a given file_id are kept in the same relative order.
        Ordering between file_ids is not, though prefix_order will return the
        order that the key was first seen.

        :param keys: An iterable of key tuples
        :return: (split_map, prefix_order)
            split_map       A dictionary mapping prefix => keys
            prefix_order    The order that we saw the various prefixes
        """
    split_by_prefix = {}
    prefix_order = []
    for key in keys:
        if len(key) == 1:
            prefix = b''
        else:
            prefix = key[0]
        if prefix in split_by_prefix:
            split_by_prefix[prefix].append(key)
        else:
            split_by_prefix[prefix] = [key]
            prefix_order.append(prefix)
    return (split_by_prefix, prefix_order)