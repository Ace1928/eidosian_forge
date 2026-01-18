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
class InterGitGitRepository(InterFromGitRepository):
    """InterRepository that copies between Git repositories."""
    source: GitRepository
    target: GitRepository

    def _get_target_either_refs(self):
        ret = {}
        for name, sha1 in self.target.controldir.get_refs_container().as_dict().items():
            ret[name] = (sha1, self.target.lookup_foreign_revision_id(sha1))
        return ret

    def fetch_refs(self, update_refs, lossy: bool=False, overwrite: bool=False) -> Tuple[RevidMap, EitherRefDict, EitherRefDict]:
        if lossy:
            raise LossyPushToSameVCS(self.source, self.target)
        old_refs = self._get_target_either_refs()
        ref_changes = {}

        def determine_wants(heads):
            old_refs = {k: (v, None) for k, v in heads.items()}
            new_refs = update_refs(old_refs)
            ret = []
            for name, (sha1, bzr_revid) in list(new_refs.items()):
                if sha1 is None:
                    sha1, unused_mapping = self.source.lookup_bzr_revision_id(bzr_revid)
                new_refs[name] = (sha1, bzr_revid)
                ret.append(sha1)
            ref_changes.update(new_refs)
            return ret
        self.fetch_objects(determine_wants)
        for k, (git_sha, bzr_revid) in ref_changes.items():
            self.target._git.refs[k] = git_sha
        new_refs = self.target.controldir.get_refs_container()
        return ({}, old_refs, new_refs)

    def fetch_objects(self, determine_wants, limit=None, mapping=None, lossy=False):
        raise NotImplementedError(self.fetch_objects)

    def _target_has_shas(self, shas):
        return {sha for sha in shas if sha in self.target._git.object_store}

    def fetch(self, revision_id=None, find_ghosts=False, fetch_spec=None, branches=None, limit=None, include_tags=False, lossy=False):
        if lossy:
            raise LossyPushToSameVCS(self.source, self.target)
        if revision_id is not None:
            args = [revision_id]
        elif fetch_spec is not None:
            recipe = fetch_spec.get_recipe()
            if recipe[0] in ('search', 'proxy-search'):
                heads = recipe[1]
            else:
                raise AssertionError('Unsupported search result type %s' % recipe[0])
            args = heads
        if branches is not None:
            determine_wants = self.get_determine_wants_branches(branches, include_tags=include_tags)
        elif fetch_spec is None and revision_id is None:
            determine_wants = self.determine_wants_all
        else:
            determine_wants = self.get_determine_wants_revids(args, include_tags=include_tags)
        wants_recorder = DetermineWantsRecorder(determine_wants)
        self.fetch_objects(wants_recorder, limit=limit)
        result = FetchResult()
        result.refs = wants_recorder.remote_refs
        return result

    def get_determine_wants_revids(self, revids, include_tags=False, tag_selector=None):
        wants = set()
        for revid in set(revids):
            if revid == NULL_REVISION:
                continue
            git_sha, mapping = self.source.lookup_bzr_revision_id(revid)
            wants.add(git_sha)
        return self.get_determine_wants_heads(wants, include_tags=include_tags, tag_selector=tag_selector)

    def get_determine_wants_branches(self, branches, include_tags=False):

        def determine_wants(refs):
            ret = []
            for name, value in refs.items():
                if value == ZERO_SHA:
                    continue
                if name.endswith(PEELED_TAG_SUFFIX):
                    continue
                if name in branches or (include_tags and is_tag(name)):
                    ret.append(value)
            return ret
        return determine_wants

    def determine_wants_all(self, refs):
        potential = {v for k, v in refs.items() if not v == ZERO_SHA and (not k.endswith(PEELED_TAG_SUFFIX))}
        return list(potential - self._target_has_shas(potential))