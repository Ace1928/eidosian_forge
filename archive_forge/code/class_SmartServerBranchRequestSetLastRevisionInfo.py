import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestSetLastRevisionInfo(SmartServerSetTipRequest):
    """Branch.set_last_revision_info.  Sets the revno and the revision ID of
    the specified branch.

    New in breezy 1.4.
    """

    def do_tip_change_with_locked_branch(self, branch, new_revno, new_last_revision_id):
        try:
            branch.set_last_revision_info(int(new_revno), new_last_revision_id)
        except errors.NoSuchRevision:
            return FailedSmartServerResponse((b'NoSuchRevision', new_last_revision_id))
        return SuccessfulSmartServerResponse((b'ok',))