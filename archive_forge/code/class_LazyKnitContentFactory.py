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
class LazyKnitContentFactory(ContentFactory):
    """A ContentFactory which can either generate full text or a wire form.

    :seealso ContentFactory:
    """

    def __init__(self, key, parents, generator, first):
        """Create a LazyKnitContentFactory.

        :param key: The key of the record.
        :param parents: The parents of the record.
        :param generator: A _ContentMapGenerator containing the record for this
            key.
        :param first: Is this the first content object returned from generator?
            if it is, its storage kind is knit-delta-closure, otherwise it is
            knit-delta-closure-ref
        """
        self.key = key
        self.parents = parents
        self.sha1 = None
        self.size = None
        self._generator = generator
        self.storage_kind = 'knit-delta-closure'
        if not first:
            self.storage_kind = self.storage_kind + '-ref'
        self._first = first

    def get_bytes_as(self, storage_kind):
        if storage_kind == self.storage_kind:
            if self._first:
                return self._generator._wire_bytes()
            else:
                return b''
        if storage_kind in ('chunked', 'fulltext', 'lines'):
            chunks = self._generator._get_one_work(self.key).text()
            if storage_kind in ('chunked', 'lines'):
                return chunks
            else:
                return b''.join(chunks)
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)

    def iter_bytes_as(self, storage_kind):
        if storage_kind in ('chunked', 'lines'):
            chunks = self._generator._get_one_work(self.key).text()
            return iter(chunks)
        raise errors.UnavailableRepresentation(self.key, storage_kind, self.storage_kind)