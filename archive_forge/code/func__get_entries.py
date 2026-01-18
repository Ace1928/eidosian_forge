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
def _get_entries(self, keys, check_present=False):
    """Get the entries for keys.

        :param keys: An iterable of index key tuples.
        """
    keys = set(keys)
    found_keys = set()
    if self._parents:
        for node in self._graph_index.iter_entries(keys):
            yield node
            found_keys.add(node[1])
    else:
        for node in self._graph_index.iter_entries(keys):
            yield (node[0], node[1], node[2], ())
            found_keys.add(node[1])
    if check_present:
        missing_keys = keys.difference(found_keys)
        if missing_keys:
            raise RevisionNotPresent(missing_keys.pop(), self)