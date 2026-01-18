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
def _parse_record_header(self, key, raw_data):
    """Parse a record header for consistency.

        :return: the header and the decompressor stream.
                 as (stream, header_record)
        """
    df = gzip.GzipFile(mode='rb', fileobj=BytesIO(raw_data))
    try:
        rec = self._check_header(key, df.readline())
    except Exception as e:
        raise KnitCorrupt(self, 'While reading {%s} got %s(%s)' % (key, e.__class__.__name__, str(e)))
    return (df, rec)