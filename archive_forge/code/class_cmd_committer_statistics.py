import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
class cmd_committer_statistics(commands.Command):
    """Generate statistics for LOCATION."""
    aliases = ['stats', 'committer-stats']
    takes_args = ['location?']
    takes_options = ['revision', option.Option('show-class', help='Show the class of contributions.')]
    encoding_type = 'replace'

    def run(self, location='.', revision=None, show_class=False):
        alternate_rev = None
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
            if len(revision) > 1:
                alternate_rev = revision[1].in_history(a_branch).rev_id
        with a_branch.lock_read():
            if alternate_rev:
                info = get_diff_info(a_branch.repository, last_rev, alternate_rev)
            else:
                info = get_info(a_branch.repository, last_rev)
        if show_class:

            def fetch_class_stats(revs):
                return gather_class_stats(a_branch.repository, revs)
        else:
            fetch_class_stats = None
        display_info(info, self.outf, fetch_class_stats)