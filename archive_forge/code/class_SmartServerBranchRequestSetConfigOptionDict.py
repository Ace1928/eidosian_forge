import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestSetConfigOptionDict(SmartServerLockedBranchRequest):
    """Set an option in the branch configuration.

    New in 2.2.
    """

    def do_with_locked_branch(self, branch, value_dict, name, section):
        utf8_dict = bencode.bdecode(value_dict)
        value_dict = {}
        for key, value in utf8_dict.items():
            value_dict[key.decode('utf8')] = value.decode('utf8')
        if not section:
            section = None
        else:
            section = section.decode('utf-8')
        branch._get_config().set_option(value_dict, name.decode('utf-8'), section)
        return SuccessfulSmartServerResponse(())