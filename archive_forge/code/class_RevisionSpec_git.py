from ..errors import InvalidRevisionId
from ..revision import NULL_REVISION
from ..revisionspec import InvalidRevisionSpec, RevisionInfo, RevisionSpec
class RevisionSpec_git(RevisionSpec):
    """Selects a revision using a Git commit SHA1."""
    help_txt = 'Selects a revision using a Git commit SHA1.\n\n    Selects a revision using a Git commit SHA1, short or long.\n\n    This works for both native Git repositories and Git revisions\n    imported into Bazaar repositories.\n    '
    prefix = 'git:'
    wants_revision_history = False

    def _lookup_git_sha1(self, branch, sha1):
        from .errors import GitSmartRemoteNotSupported
        from .mapping import default_mapping
        bzr_revid = getattr(branch.repository, 'lookup_foreign_revision_id', default_mapping.revision_id_foreign_to_bzr)(sha1)
        try:
            if branch.repository.has_revision(bzr_revid):
                return bzr_revid
        except GitSmartRemoteNotSupported:
            return bzr_revid
        raise InvalidRevisionSpec(self.user_spec, branch)

    def __nonzero__(self):
        if self.rev_id is None:
            return False
        if self.rev_id == NULL_REVISION:
            return False
        return True

    def _find_short_git_sha1(self, branch, sha1):
        from .mapping import ForeignGit, mapping_registry
        parse_revid = getattr(branch.repository, 'lookup_bzr_revision_id', mapping_registry.parse_revision_id)

        def matches_revid(revid):
            if revid == NULL_REVISION:
                return False
            try:
                foreign_revid, mapping = parse_revid(revid)
            except InvalidRevisionId:
                return False
            if not isinstance(mapping.vcs, ForeignGit):
                return False
            return foreign_revid.startswith(sha1)
        with branch.repository.lock_read():
            graph = branch.repository.get_graph()
            last_revid = branch.last_revision()
            if matches_revid(last_revid):
                return last_revid
            for revid, _ in graph.iter_ancestry([last_revid]):
                if matches_revid(revid):
                    return revid
            raise InvalidRevisionSpec(self.user_spec, branch)

    def _as_revision_id(self, context_branch):
        loc = self.spec.find(':')
        git_sha1 = self.spec[loc + 1:].encode('utf-8')
        if len(git_sha1) > 40 or len(git_sha1) < 4 or (not valid_git_sha1(git_sha1)):
            raise InvalidRevisionSpec(self.user_spec, context_branch)
        from . import lazy_check_versions
        lazy_check_versions()
        if len(git_sha1) == 40:
            return self._lookup_git_sha1(context_branch, git_sha1)
        else:
            return self._find_short_git_sha1(context_branch, git_sha1)

    def _match_on(self, branch, revs):
        revid = self._as_revision_id(branch)
        return RevisionInfo.from_revision_id(branch, revid)

    def needs_branch(self):
        return True

    def get_branch(self):
        return None