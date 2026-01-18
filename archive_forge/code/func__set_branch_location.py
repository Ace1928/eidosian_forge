from . import errors, lock, merge, revision
from .branch import Branch
from .i18n import gettext
from .trace import note
def _set_branch_location(control, to_branch, current_branch, force=False):
    """Set location value of a branch reference.

    :param control: ControlDir of the checkout to change
    :param to_branch: branch that the checkout is to reference
    :param force: skip the check for local commits in a heavy checkout
    """
    branch_format = control.find_branch_format()
    if branch_format.get_reference(control) is not None:
        branch_format.set_reference(control, None, to_branch)
    else:
        b = current_branch
        bound_branch = b.get_bound_location()
        if bound_branch is not None:
            possible_transports = []
            try:
                if not force and _any_local_commits(b, possible_transports):
                    raise errors.CommandError(gettext('Cannot switch as local commits found in the checkout. Commit these to the bound branch or use --force to throw them away.'))
            except errors.BoundBranchConnectionFailure as e:
                raise errors.CommandError(gettext('Unable to connect to current master branch %(target)s: %(error)s To switch anyway, use --force.') % e.__dict__)
            with b.lock_write():
                b.set_bound_location(None)
                b.pull(to_branch, overwrite=True, possible_transports=possible_transports)
                b.set_bound_location(to_branch.base)
                b.set_parent(b.get_master_branch().get_parent())
        else:
            with b.lock_read():
                graph = b.repository.get_graph(to_branch.repository)
                if b.controldir._format.colocated_branches and (force or graph.is_ancestor(b.last_revision(), to_branch.last_revision())):
                    b.controldir.destroy_branch()
                    b.controldir.set_branch_reference(to_branch, name='')
                else:
                    raise errors.CommandError(gettext('Cannot switch a branch, only a checkout.'))