import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestSetLastRevision(SmartServerSetTipRequest):

    def do_tip_change_with_locked_branch(self, branch, new_last_revision_id):
        if new_last_revision_id == b'null:':
            branch.set_last_revision_info(0, new_last_revision_id)
        else:
            if not branch.repository.has_revision(new_last_revision_id):
                return FailedSmartServerResponse((b'NoSuchRevision', new_last_revision_id))
            branch.generate_revision_history(new_last_revision_id, None, None)
        return SuccessfulSmartServerResponse((b'ok',))