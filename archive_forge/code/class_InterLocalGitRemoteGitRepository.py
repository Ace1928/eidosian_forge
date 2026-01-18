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
class InterLocalGitRemoteGitRepository(InterToGitRepository):

    def fetch_refs(self, update_refs, lossy=False, overwrite=False):
        """Import the gist of the ancestry of a particular revision."""
        if lossy:
            raise LossyPushToSameVCS(self.source, self.target)

        def git_update_refs(old_refs):
            ret = {}
            self.old_refs = {k: (v, None) for k, v in old_refs.items()}
            new_refs = update_refs(self.old_refs)
            for name, (gitid, revid) in new_refs.items():
                if gitid is None:
                    gitid = self.source_store._lookup_revision_sha1(revid)
                if not overwrite:
                    if remote_divergence(old_refs.get(name), gitid, self.source_store):
                        raise DivergedBranches(self.source, self.target)
                ret[name] = gitid
            return ret
        new_refs = self.target.send_pack(git_update_refs, self.source._git.generate_pack_data)
        return (None, self.old_refs, new_refs)

    @staticmethod
    def is_compatible(source, target):
        return isinstance(source, LocalGitRepository) and isinstance(target, RemoteGitRepository)