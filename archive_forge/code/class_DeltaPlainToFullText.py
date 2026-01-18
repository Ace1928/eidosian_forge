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
class DeltaPlainToFullText(KnitAdapter):
    """An adapter for deltas from annotated to unannotated."""

    def get_bytes(self, factory, target_storage_kind):
        compressed_bytes = factory._raw_record
        rec, contents = self._data._parse_record_unchecked(compressed_bytes)
        delta = self._plain_factory.parse_line_delta(contents, rec[1])
        compression_parent = factory.parents[0]
        basis_entry = next(self._basis_vf.get_record_stream([compression_parent], 'unordered', True))
        if basis_entry.storage_kind == 'absent':
            raise errors.RevisionNotPresent(compression_parent, self._basis_vf)
        basis_lines = basis_entry.get_bytes_as('lines')
        basis_content = PlainKnitContent(basis_lines, compression_parent)
        content, _ = self._plain_factory.parse_record(rec[1], contents, factory._build_details, basis_content)
        if target_storage_kind == 'fulltext':
            return b''.join(content.text())
        elif target_storage_kind in ('chunked', 'lines'):
            return content.text()
        raise UnavailableRepresentation(factory.key, target_storage_kind, factory.storage_kind)