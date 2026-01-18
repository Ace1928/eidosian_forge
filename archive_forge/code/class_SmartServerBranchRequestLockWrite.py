import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestLockWrite(SmartServerBranchRequest):

    def do_with_branch(self, branch, branch_token=b'', repo_token=b''):
        if branch_token == b'':
            branch_token = None
        if repo_token == b'':
            repo_token = None
        try:
            repo_token = branch.repository.lock_write(token=repo_token).repository_token
            try:
                branch_token = branch.lock_write(token=branch_token).token
            finally:
                branch.repository.unlock()
        except errors.LockContention:
            return FailedSmartServerResponse((b'LockContention',))
        except errors.TokenMismatch:
            return FailedSmartServerResponse((b'TokenMismatch',))
        except errors.UnlockableTransport:
            return FailedSmartServerResponse((b'UnlockableTransport',))
        except errors.LockFailed as e:
            return FailedSmartServerResponse((b'LockFailed', str(e.lock).encode('utf-8'), str(e.why).encode('utf-8')))
        if repo_token is None:
            repo_token = b''
        else:
            branch.repository.leave_lock_in_place()
        branch.leave_lock_in_place()
        branch.unlock()
        return SuccessfulSmartServerResponse((b'ok', branch_token, repo_token))