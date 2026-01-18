import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestUnlock(SmartServerBranchRequest):

    def do_with_branch(self, branch, branch_token, repo_token):
        try:
            with branch.repository.lock_write(token=repo_token):
                branch.lock_write(token=branch_token)
        except errors.TokenMismatch:
            return FailedSmartServerResponse((b'TokenMismatch',))
        if repo_token:
            branch.repository.dont_leave_lock_in_place()
        branch.dont_leave_lock_in_place()
        branch.unlock()
        return SuccessfulSmartServerResponse((b'ok',))