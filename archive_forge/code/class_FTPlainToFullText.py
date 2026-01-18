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
class FTPlainToFullText(KnitAdapter):
    """An adapter from FT plain knits to unannotated ones."""

    def get_bytes(self, factory, target_storage_kind):
        compressed_bytes = factory._raw_record
        rec, contents = self._data._parse_record_unchecked(compressed_bytes)
        content, delta = self._plain_factory.parse_record(factory.key[-1], contents, factory._build_details, None)
        if target_storage_kind == 'fulltext':
            return b''.join(content.text())
        elif target_storage_kind in ('chunked', 'lines'):
            return content.text()
        raise UnavailableRepresentation(factory.key, target_storage_kind, factory.storage_kind)