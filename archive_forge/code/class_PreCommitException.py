from breezy import branch, delta, errors, revision, transport
from breezy.tests import per_branch
class PreCommitException(Exception):

    def __init__(self, revid):
        self.revid = revid