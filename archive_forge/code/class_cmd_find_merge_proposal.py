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
class cmd_find_merge_proposal(Command):
    __doc__ = 'Find a merge proposal.\n\n    '
    takes_options = ['directory']
    takes_args = ['submit_branch?']
    aliases = ['find-proposal']

    def run(self, directory='.', submit_branch=None):
        tree, branch, relpath = controldir.ControlDir.open_containing_tree_or_branch(directory)
        public_location = branch.get_public_branch()
        if public_location:
            branch = _mod_branch.Branch.open(public_location)
        if submit_branch is None:
            submit_branch = branch.get_submit_branch()
        if submit_branch is None:
            submit_branch = branch.get_parent()
        if submit_branch is None:
            raise errors.CommandError(gettext('No target location specified or remembered'))
        else:
            target = _mod_branch.Branch.open(submit_branch)
        forge = _mod_forge.get_forge(branch)
        for mp in forge.iter_proposals(branch, target):
            self.outf.write(gettext('Merge proposal: %s\n') % mp.url)