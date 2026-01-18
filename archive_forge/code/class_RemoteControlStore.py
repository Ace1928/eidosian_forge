import bz2
import os
import re
import sys
import zlib
from typing import Callable, List, Optional
import fastbencode as bencode
from .. import branch
from .. import bzr as _mod_bzr
from .. import config as _mod_config
from .. import (controldir, debug, errors, gpg, graph, lock, lockdir, osutils,
from .. import repository as _mod_repository
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..branch import BranchWriteLockResult
from ..decorators import only_raises
from ..errors import NoSuchRevision, SmartProtocolError
from ..i18n import gettext
from ..lockable_files import LockableFiles
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..revision import NULL_REVISION
from ..trace import log_exception_quietly, mutter, note, warning
from . import branch as bzrbranch
from . import bzrdir as _mod_bzrdir
from . import inventory_delta
from . import repository as bzrrepository
from . import testament as _mod_testament
from . import vf_repository, vf_search
from .branch import BranchReferenceFormat
from .inventory import Inventory
from .inventorytree import InventoryRevisionTree
from .serializer import format_registry as serializer_format_registry
from .smart import client
from .smart import repository as smart_repo
from .smart import vfs
from .smart.client import _SmartClient
from .versionedfile import FulltextContentFactory
class RemoteControlStore(_mod_config.IniFileStore):
    """Control store which attempts to use HPSS calls to retrieve control store.

    Note that this is specific to bzr-based formats.
    """

    def __init__(self, bzrdir):
        super().__init__()
        self.controldir = bzrdir
        self._real_store = None

    def lock_write(self, token=None):
        self._ensure_real()
        return self._real_store.lock_write(token)

    def unlock(self):
        self._ensure_real()
        return self._real_store.unlock()

    def save(self):
        with self.lock_write():
            self.save_without_locking()

    def save_without_locking(self):
        super().save()

    def _ensure_real(self):
        self.controldir._ensure_real()
        if self._real_store is None:
            self._real_store = _mod_config.ControlStore(self.controldir)

    def external_url(self):
        return urlutils.join(self.branch.user_url, 'control.conf')

    def _load_content(self):
        medium = self.controldir._client._medium
        path = self.controldir._path_for_remote_call(self.controldir._client)
        try:
            response, handler = self.controldir._call_expecting_body(b'BzrDir.get_config_file', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_store._load_content()
        if len(response) and response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        return handler.read_body_bytes()

    def _save_content(self, content):
        self._ensure_real()
        return self._real_store._save_content(content)