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
def _ensure_real(self):
    """Ensure that there is a _real_branch set.

        Used before calls to self._real_branch.
        """
    if self._real_branch is None:
        if not vfs.vfs_enabled():
            raise AssertionError('smart server vfs must be enabled to use vfs implementation')
        self.controldir._ensure_real()
        self._real_branch = self.controldir._real_bzrdir.open_branch(ignore_fallbacks=self._real_ignore_fallbacks, name=self._name)
        self._real_branch.conf_store = self.conf_store
        if self.repository._real_repository is None:
            real_repo = self._real_branch.repository
            if isinstance(real_repo, RemoteRepository):
                real_repo._ensure_real()
                real_repo = real_repo._real_repository
            self.repository._set_real_repository(real_repo)
        self._real_branch.repository = self.repository
        if self._lock_mode == 'r':
            self._real_branch.lock_read()
        elif self._lock_mode == 'w':
            self._real_branch.lock_write(token=self._lock_token)