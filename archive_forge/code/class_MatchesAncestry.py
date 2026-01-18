from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
class MatchesAncestry(Matcher):
    """A matcher that checks the ancestry of a particular revision.

    :ivar graph: Graph in which to check the ancestry
    :ivar revision_id: Revision id of the revision
    """

    def __init__(self, repository, revision_id):
        Matcher.__init__(self)
        self.repository = repository
        self.revision_id = revision_id

    def __str__(self):
        return 'MatchesAncestry(repository={!r}, revision_id={!r})'.format(self.repository, self.revision_id)

    def match(self, expected):
        with self.repository.lock_read():
            graph = self.repository.get_graph()
            got = [r for r, p in graph.iter_ancestry([self.revision_id])]
            if _mod_revision.NULL_REVISION in got:
                got.remove(_mod_revision.NULL_REVISION)
        if sorted(got) != sorted(expected):
            return _AncestryMismatch(self.revision_id, sorted(got), sorted(expected))