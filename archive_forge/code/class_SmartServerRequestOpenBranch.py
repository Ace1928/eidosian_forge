import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestOpenBranch(SmartServerRequestBzrDir):

    def do_bzrdir_request(self):
        """open a branch at path and return the branch reference or branch."""
        try:
            reference_url = self._bzrdir.get_branch_reference()
            if reference_url is None:
                reference_url = ''
            return SuccessfulSmartServerResponse((b'ok', reference_url.encode('utf-8')))
        except errors.NotBranchError as e:
            return FailedSmartServerResponse((b'nobranch',))