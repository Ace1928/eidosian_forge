import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestOpenBzrDir_2_1(SmartServerRequest):

    def do(self, path):
        """Is there a BzrDir present, and if so does it have a working tree?

        New in 2.1.
        """
        try:
            t = self.transport_from_client_path(path)
        except errors.PathNotChild:
            return SuccessfulSmartServerResponse((b'no',))
        try:
            bd = BzrDir.open_from_transport(t)
        except errors.NotBranchError:
            answer = (b'no',)
        else:
            answer = (b'yes',)
            if bd.has_workingtree():
                answer += (b'yes',)
            else:
                answer += (b'no',)
        return SuccessfulSmartServerResponse(answer)