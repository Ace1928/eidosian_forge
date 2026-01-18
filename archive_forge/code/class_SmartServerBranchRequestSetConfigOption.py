import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestSetConfigOption(SmartServerLockedBranchRequest):
    """Set an option in the branch configuration."""

    def do_with_locked_branch(self, branch, value, name, section):
        if not section:
            section = None
        branch._get_config().set_option(value.decode('utf-8'), name.decode('utf-8'), section.decode('utf-8') if section is not None else None)
        return SuccessfulSmartServerResponse(())