from ...commands import Command, display_command
from ...errors import (CommandError, ConflictsInTree, NoWorkingTree,
from ...i18n import gettext
from ...option import Option
from ...trace import note
from ...transport import NoSuchFile
class cmd_rebase_abort(Command):
    """Abort an interrupted rebase."""
    takes_options = [Option('directory', short_name='d', help='Branch to replay onto, rather than the one containing the working directory.', type=str)]

    @display_command
    def run(self, directory='.'):
        from ...workingtree import WorkingTree
        from .rebase import RebaseState1, complete_revert
        wt = WorkingTree.open_containing(directory)[0]
        wt.lock_write()
        try:
            state = RebaseState1(wt)
            try:
                last_rev_info = state.read_plan()[0]
            except NoSuchFile:
                raise CommandError('No rebase to abort')
            complete_revert(wt, [last_rev_info[1]])
            state.remove_plan()
        finally:
            wt.unlock()