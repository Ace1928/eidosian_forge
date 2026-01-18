from .. import debug, errors
from .. import revision as _mod_revision
from ..branch import Branch
from ..trace import mutter_callsite
from .branch import BranchFormatMetadir, BzrBranch
def _write_revision_history(self, history):
    """Factored out of set_revision_history.

        This performs the actual writing to disk.
        It is intended to be called by set_revision_history."""
    self._transport.put_bytes('revision-history', b'\n'.join(history), mode=self.controldir._get_file_mode())