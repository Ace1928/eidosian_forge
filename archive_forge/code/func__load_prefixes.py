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
def _load_prefixes(self, prefixes):
    """Load the indices for prefixes."""
    self._check_read()
    for prefix in prefixes:
        if prefix not in self._kndx_cache:
            self._cache = {}
            self._history = []
            self._filename = prefix
            try:
                path = self._mapper.map(prefix) + '.kndx'
                with self._transport.get(path) as fp:
                    _load_data(self, fp)
                self._kndx_cache[prefix] = (self._cache, self._history)
                del self._cache
                del self._filename
                del self._history
            except NoSuchFile:
                self._kndx_cache[prefix] = ({}, [])
                if isinstance(self._mapper, ConstantMapper):
                    self._init_index(path)
                del self._cache
                del self._filename
                del self._history