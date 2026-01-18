from ...commands import Command, display_command
from ...errors import (CommandError, ConflictsInTree, NoWorkingTree,
from ...i18n import gettext
from ...option import Option
from ...trace import note
from ...transport import NoSuchFile
class cmd_rebase_continue(Command):
    """Continue an interrupted rebase after resolving conflicts."""
    takes_options = ['merge-type', Option('directory', short_name='d', help='Branch to replay onto, rather than the one containing the working directory.', type=str)]

    @display_command
    def run(self, merge_type=None, directory='.'):
        from ...workingtree import WorkingTree
        from .rebase import RebaseState1, WorkingTreeRevisionRewriter
        wt = WorkingTree.open_containing(directory)[0]
        wt.lock_write()
        try:
            state = RebaseState1(wt)
            replayer = WorkingTreeRevisionRewriter(wt, state, merge_type=merge_type)
            if len(wt.conflicts()) != 0:
                raise CommandError(gettext("There are still conflicts present. Resolve the conflicts and then run 'brz resolve' and try again."))
            try:
                replace_map = state.read_plan()[1]
            except NoSuchFile:
                raise CommandError(gettext('No rebase to continue'))
            oldrevid = state.read_active_revid()
            if oldrevid is not None:
                oldrev = wt.branch.repository.get_revision(oldrevid)
                replayer.commit_rebase(oldrev, replace_map[oldrevid][0])
            finish_rebase(state, wt, replace_map, replayer)
        finally:
            wt.unlock()