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
def _get_record_map_unparsed(self, keys, allow_missing=False):
    """Get the raw data for reconstructing keys without parsing it.

        :return: A dict suitable for parsing via _raw_map_to_record_map.
            key-> raw_bytes, (method, noeol), compression_parent
        """
    while True:
        try:
            position_map = self._get_components_positions(keys, allow_missing=allow_missing)
            records = [(key, i_m) for key, (r, i_m, n) in position_map.items()]
            records.sort(key=operator.itemgetter(1))
            raw_record_map = {}
            for key, data in self._read_records_iter_unchecked(records):
                record_details, index_memo, next = position_map[key]
                raw_record_map[key] = (data, record_details, next)
            return raw_record_map
        except pack_repo.RetryWithNewPacks as e:
            self._access.reload_or_raise(e)