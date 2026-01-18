import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestGetPhysicalLockStatus(SmartServerBranchRequest):
    """Get the physical lock status for a branch.

    New in 2.5.
    """

    def do_with_branch(self, branch):
        if branch.get_physical_lock_status():
            return SuccessfulSmartServerResponse((b'yes',))
        else:
            return SuccessfulSmartServerResponse((b'no',))