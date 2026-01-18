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
class KnitContentFactory(ContentFactory):
    """Content factory for streaming from knits.

    :seealso ContentFactory:
    """

    def __init__(self, key, parents, build_details, sha1, raw_record, annotated, knit=None, network_bytes=None):
        """Create a KnitContentFactory for key.

        :param key: The key.
        :param parents: The parents.
        :param build_details: The build details as returned from
            get_build_details.
        :param sha1: The sha1 expected from the full text of this object.
        :param raw_record: The bytes of the knit data from disk.
        :param annotated: True if the raw data is annotated.
        :param network_bytes: None to calculate the network bytes on demand,
            not-none if they are already known.
        """
        ContentFactory.__init__(self)
        self.sha1 = sha1
        self.key = key
        self.parents = parents
        if build_details[0] == 'line-delta':
            kind = 'delta'
        else:
            kind = 'ft'
        if annotated:
            annotated_kind = 'annotated-'
        else:
            annotated_kind = ''
        self.storage_kind = 'knit-{}{}-gz'.format(annotated_kind, kind)
        self._raw_record = raw_record
        self._network_bytes = network_bytes
        self._build_details = build_details
        self._knit = knit

    def _create_network_bytes(self):
        """Create a fully serialised network version for transmission."""
        key_bytes = b'\x00'.join(self.key)
        if self.parents is None:
            parent_bytes = b'None:'
        else:
            parent_bytes = b'\t'.join((b'\x00'.join(key) for key in self.parents))
        if self._build_details[1]:
            noeol = b'N'
        else:
            noeol = b' '
        network_bytes = b'%s\n%s\n%s\n%s%s' % (self.storage_kind.encode('ascii'), key_bytes, parent_bytes, noeol, self._raw_record)
        self._network_bytes = network_bytes

    def get_bytes_as(self, storage_kind):
        if storage_kind == self.storage_kind:
            if self._network_bytes is None:
                self._create_network_bytes()
            return self._network_bytes
        if '-ft-' in self.storage_kind and storage_kind in ('chunked', 'fulltext', 'lines'):
            adapter_key = (self.storage_kind, storage_kind)
            adapter_factory = adapter_registry.get(adapter_key)
            adapter = adapter_factory(None)
            return adapter.get_bytes(self, storage_kind)
        if self._knit is not None:
            if storage_kind in ('chunked', 'lines'):
                return self._knit.get_lines(self.key[0])
            elif storage_kind == 'fulltext':
                return self._knit.get_text(self.key[0])
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)

    def iter_bytes_as(self, storage_kind):
        return iter(self.get_bytes_as(storage_kind))