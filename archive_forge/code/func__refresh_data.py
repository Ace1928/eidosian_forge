from typing import Type
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import transport as _mod_transport
from ..repository import InterRepository, IsInWriteGroupError, Repository
from .repository import RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (InterSameDataRepository,
def _refresh_data(self):
    if not self.is_locked():
        return
    if self.is_in_write_group():
        raise IsInWriteGroupError(self)
    self.control_files._finish_transaction()
    if self.is_write_locked():
        self.control_files._set_write_transaction()
    else:
        self.control_files._set_read_transaction()