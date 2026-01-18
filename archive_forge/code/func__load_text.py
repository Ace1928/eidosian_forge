import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
def _load_text(self, key):
    if not self._is_locked():
        raise errors.ObjectNotLocked(self)
    path = self._map(key)
    try:
        text = self._transport.get_bytes(path)
        compressed = self._compressed
    except _mod_transport.NoSuchFile:
        if self._compressed:
            path = path[:-3]
            try:
                text = self._transport.get_bytes(path)
                compressed = False
            except _mod_transport.NoSuchFile:
                return None
        else:
            return None
    if compressed:
        text = gzip.GzipFile(mode='rb', fileobj=BytesIO(text)).read()
    return text