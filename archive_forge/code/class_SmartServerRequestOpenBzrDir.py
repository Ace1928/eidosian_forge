import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestOpenBzrDir(SmartServerRequest):

    def do(self, path):
        try:
            t = self.transport_from_client_path(path)
        except errors.PathNotChild:
            answer = b'no'
        else:
            bzr_prober = BzrProber()
            try:
                bzr_prober.probe_transport(t)
            except (errors.NotBranchError, errors.UnknownFormatError):
                answer = b'no'
            else:
                answer = b'yes'
        return SuccessfulSmartServerResponse((answer,))