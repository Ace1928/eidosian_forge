from io import StringIO
from ... import branch as _mod_branch
from ... import controldir, errors
from ... import forge as _mod_forge
from ... import log as _mod_log
from ... import missing as _mod_missing
from ... import msgeditor, urlutils
from ...commands import Command
from ...i18n import gettext
from ...option import ListOption, Option, RegistryOption
from ...trace import note, warning
class cmd_publish_derived(Command):
    __doc__ = 'Publish a derived branch.\n\n    Try to create a public copy of a local branch on a hosting site,\n    derived from the specified base branch.\n\n    Reasonable defaults are picked for owner name, branch name and project\n    name, but they can also be overridden from the command-line.\n    '
    takes_options = ['directory', Option('owner', help='Owner of the new remote branch.', type=str), Option('project', help='Project name for the new remote branch.', type=str), Option('name', help='Name of the new remote branch.', type=str), Option('no-allow-lossy', help='Allow fallback to lossy push, if necessary.'), Option('overwrite', help='Overwrite existing commits.'), 'revision']
    takes_args = ['submit_branch?']

    def run(self, submit_branch=None, owner=None, name=None, project=None, no_allow_lossy=False, overwrite=False, directory='.', revision=None):
        local_branch = _mod_branch.Branch.open_containing(directory)[0]
        self.add_cleanup(local_branch.lock_write().unlock)
        if submit_branch is None:
            submit_branch = local_branch.get_submit_branch()
            note(gettext('Using submit branch %s') % submit_branch)
        if submit_branch is None:
            submit_branch = local_branch.get_parent()
            note(gettext('Using parent branch %s') % submit_branch)
        submit_branch = _mod_branch.Branch.open(submit_branch)
        _check_already_merged(local_branch, submit_branch)
        if name is None:
            name = branch_name(local_branch)
        forge = _mod_forge.get_forge(submit_branch)
        if revision is None:
            stop_revision = None
        else:
            stop_revision = revision.as_revision_id(branch)
        remote_branch, public_url = forge.publish_derived(local_branch, submit_branch, name=name, project=project, owner=owner, allow_lossy=not no_allow_lossy, overwrite=overwrite, revision_id=stop_revision)
        local_branch.set_push_location(remote_branch.user_url)
        local_branch.set_public_branch(public_url)
        local_branch.set_submit_branch(submit_branch.user_url)
        note(gettext('Pushed to %s') % public_url)