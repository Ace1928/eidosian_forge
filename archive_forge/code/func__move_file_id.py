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
def _move_file_id(self, from_id, to_id):
    t = self._transport.clone('knits')
    from_rel_url = self.texts._index._mapper.map((from_id, None))
    to_rel_url = self.texts._index._mapper.map((to_id, None))
    for suffix in ('.knit', '.kndx'):
        t.rename(from_rel_url + suffix, to_rel_url + suffix)