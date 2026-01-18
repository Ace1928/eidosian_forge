import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestOpenBranchV2(SmartServerRequestBzrDir):

    def do_bzrdir_request(self):
        """open a branch at path and return the reference or format."""
        try:
            reference_url = self._bzrdir.get_branch_reference()
            if reference_url is None:
                br = self._bzrdir.open_branch(ignore_fallbacks=True)
                format = br._format.network_name()
                return SuccessfulSmartServerResponse((b'branch', format))
            else:
                return SuccessfulSmartServerResponse((b'ref', reference_url.encode('utf-8')))
        except errors.NotBranchError as e:
            return FailedSmartServerResponse((b'nobranch',))