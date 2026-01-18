from . import errors, lock, merge, revision
from .branch import Branch
from .i18n import gettext
from .trace import note
def _any_local_commits(this_branch, possible_transports):
    """Does this branch have any commits not in the master branch?"""
    last_rev = this_branch.last_revision()
    if last_rev != revision.NULL_REVISION:
        other_branch = this_branch.get_master_branch(possible_transports)
        with this_branch.lock_read(), other_branch.lock_read():
            other_last_rev = other_branch.last_revision()
            graph = this_branch.repository.get_graph(other_branch.repository)
            if not graph.is_ancestor(last_rev, other_last_rev):
                return True
    return False