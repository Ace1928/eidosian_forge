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
def _set_config_option(self, value, name, section):
    if isinstance(value, (bool, int)):
        value = str(value)
    elif isinstance(value, str):
        pass
    else:
        raise TypeError(value)
    try:
        path = self._branch._remote_path()
        response = self._branch._client.call(b'Branch.set_config_option', path, self._branch._lock_token, self._branch._repo_lock_token, value.encode('utf-8'), name.encode('utf-8'), (section or '').encode('utf-8'))
    except errors.UnknownSmartMethod:
        medium = self._branch._client._medium
        medium._remember_remote_is_before((1, 14))
        return self._vfs_set_option(value, name, section)
    if response != ():
        raise errors.UnexpectedSmartServerResponse(response)