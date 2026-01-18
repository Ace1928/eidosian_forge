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
def _read_records_iter(self, records):
    """Read text records from data file and yield result.

        The result will be returned in whatever is the fastest to read.
        Not by the order requested. Also, multiple requests for the same
        record will only yield 1 response.

        :param records: A list of (key, access_memo) entries
        :return: Yields (key, contents, digest) in the order
                 read, not the order requested
        """
    if not records:
        return
    needed_records = sorted(set(records), key=operator.itemgetter(1))
    if not needed_records:
        return
    raw_data = self._access.get_raw_records([index_memo for key, index_memo in needed_records])
    for (key, index_memo), data in zip(needed_records, raw_data):
        content, digest = self._parse_record(key[-1], data)
        yield (key, content, digest)