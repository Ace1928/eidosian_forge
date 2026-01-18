import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
class cmd_credits(commands.Command):
    """Determine credits for LOCATION."""
    takes_args = ['location?']
    takes_options = ['revision']
    encoding_type = 'replace'

    def run(self, location='.', revision=None):
        try:
            wt = workingtree.WorkingTree.open_containing(location)[0]
        except errors.NoWorkingTree:
            a_branch = branch.Branch.open(location)
            last_rev = a_branch.last_revision()
        else:
            a_branch = wt.branch
            last_rev = wt.last_revision()
        if revision is not None:
            last_rev = revision[0].in_history(a_branch).rev_id
        with a_branch.lock_read():
            credits = find_credits(a_branch.repository, last_rev)
            display_credits(credits, self.outf)