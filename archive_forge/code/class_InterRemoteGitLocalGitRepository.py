import itertools
from typing import Callable, Dict, Tuple, Optional
from dulwich.errors import NotCommitError
from dulwich.objects import ObjectID
from dulwich.object_store import ObjectStoreGraphWalker
from dulwich.pack import PACK_SPOOL_FILE_MAX_SIZE
from dulwich.protocol import CAPABILITY_THIN_PACK, ZERO_SHA
from dulwich.refs import SYMREF
from dulwich.walk import Walker
from .. import config, trace, ui
from ..errors import (DivergedBranches, FetchLimitUnsupported,
from ..repository import FetchResult, InterRepository, AbstractSearchResult
from ..revision import NULL_REVISION, RevisionID
from .errors import NoPushSupport
from .fetch import DetermineWantsRecorder, import_git_objects
from .mapping import needs_roundtripping
from .object_store import get_object_store
from .push import MissingObjectsIterator, remote_divergence
from .refs import is_tag, ref_to_tag_name
from .remote import RemoteGitError, RemoteGitRepository
from .repository import GitRepository, GitRepositoryFormat, LocalGitRepository
from .unpeel_map import UnpeelMap
class InterRemoteGitLocalGitRepository(InterGitGitRepository):

    def fetch_objects(self, determine_wants, limit=None, mapping=None):
        from tempfile import SpooledTemporaryFile
        if limit is not None:
            raise FetchLimitUnsupported(self)
        graphwalker = self.target._git.get_graph_walker()
        if CAPABILITY_THIN_PACK in self.source.controldir._client._fetch_capabilities:
            f = SpooledTemporaryFile(max_size=PACK_SPOOL_FILE_MAX_SIZE, prefix='incoming-', dir=getattr(self.target._git.object_store, 'path', None))

            def commit():
                if f.tell():
                    f.seek(0)
                    self.target._git.object_store.add_thin_pack(f.read, None)

            def abort():
                pass
        else:
            f, commit, abort = self.target._git.object_store.add_pack()
        try:
            refs = self.source.controldir.fetch_pack(determine_wants, graphwalker, f.write)
            commit()
            return (None, None, refs)
        except BaseException:
            abort()
            raise

    @staticmethod
    def is_compatible(source, target):
        """Be compatible with GitRepository."""
        return isinstance(source, RemoteGitRepository) and isinstance(target, LocalGitRepository)