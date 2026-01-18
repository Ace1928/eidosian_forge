import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestRevisionHistory(SmartServerBranchRequest):

    def do_with_branch(self, branch):
        """Get the revision history for the branch.

        The revision list is returned as the body content,
        with each revision utf8 encoded and \x00 joined.
        """
        with branch.lock_read():
            graph = branch.repository.get_graph()
            stop_revisions = (None, _mod_revision.NULL_REVISION)
            history = list(graph.iter_lefthand_ancestry(branch.last_revision(), stop_revisions))
        return SuccessfulSmartServerResponse((b'ok',), b'\x00'.join(reversed(history)))