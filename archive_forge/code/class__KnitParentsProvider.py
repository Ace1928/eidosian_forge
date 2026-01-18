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
class _KnitParentsProvider:

    def __init__(self, knit):
        self._knit = knit

    def __repr__(self):
        return 'KnitParentsProvider(%r)' % self._knit

    def get_parent_map(self, keys):
        """See graph.StackedParentsProvider.get_parent_map"""
        parent_map = {}
        for revision_id in keys:
            if revision_id is None:
                raise ValueError('get_parent_map(None) is not valid')
            if revision_id == _mod_revision.NULL_REVISION:
                parent_map[revision_id] = ()
            else:
                try:
                    parents = tuple(self._knit.get_parents_with_ghosts(revision_id))
                except errors.RevisionNotPresent:
                    continue
                else:
                    if len(parents) == 0:
                        parents = (_mod_revision.NULL_REVISION,)
                parent_map[revision_id] = parents
        return parent_map