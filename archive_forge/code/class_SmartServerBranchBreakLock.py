import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchBreakLock(SmartServerBranchRequest):

    def do_with_branch(self, branch):
        """Break a branch lock.
        """
        branch.break_lock()
        return SuccessfulSmartServerResponse((b'ok',))