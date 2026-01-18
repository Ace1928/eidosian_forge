import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchGetTagsBytes(SmartServerBranchRequest):

    def do_with_branch(self, branch):
        """Return the _get_tags_bytes for a branch."""
        bytes = branch._get_tags_bytes()
        return SuccessfulSmartServerResponse((bytes,))