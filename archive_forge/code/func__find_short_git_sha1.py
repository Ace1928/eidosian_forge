from ..errors import InvalidRevisionId
from ..revision import NULL_REVISION
from ..revisionspec import InvalidRevisionSpec, RevisionInfo, RevisionSpec
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