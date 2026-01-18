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
class SignatureTextStore(TextVersionedFiles):
    """Legacy thunk for format 4-7 repositories."""

    def __init__(self, transport, compressed, mapper, is_locked, can_write):
        TextVersionedFiles.__init__(self, transport, compressed, mapper, is_locked, can_write)
        self._ext = '.sig' + self._ext

    def get_parent_map(self, keys):
        result = {}
        for key in keys:
            text = self._load_text(key)
            if text is None:
                continue
            result[key] = None
        return result

    def get_record_stream(self, keys, sort_order, include_delta_closure):
        for key in keys:
            text = self._load_text(key)
            if text is None:
                yield AbsentContentFactory(key)
            else:
                yield FulltextContentFactory(key, None, None, text)

    def keys(self):
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)
        relpaths = set()
        for quoted_relpath in self._transport.iter_files_recursive():
            relpath = urlutils.unquote(quoted_relpath)
            path, ext = os.path.splitext(relpath)
            if ext == '.gz':
                relpath = path
            if not relpath.endswith('.sig'):
                continue
            relpaths.add(relpath[:-4])
        paths = list(relpaths)
        return {self._mapper.unmap(path) for path in paths}