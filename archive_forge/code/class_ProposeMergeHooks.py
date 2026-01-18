import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class ProposeMergeHooks(hooks.Hooks):
    """Hooks for proposing a merge on Launchpad."""

    def __init__(self):
        hooks.Hooks.__init__(self, __name__, 'Proposer.hooks')
        self.add_hook('get_prerequisite', 'Return the prerequisite branch for proposing as merge.', (3, 0))
        self.add_hook('merge_proposal_body', 'Return an initial body for the merge proposal message.', (3, 0))