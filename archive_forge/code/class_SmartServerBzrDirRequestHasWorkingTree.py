import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBzrDirRequestHasWorkingTree(SmartServerRequestBzrDir):

    def do_bzrdir_request(self, name=None):
        """Check whether there is a working tree present.

        New in 2.5.0.

        :return: If there is a working tree present, 'yes'.
            Otherwise 'no'.
        """
        if self._bzrdir.has_workingtree():
            return SuccessfulSmartServerResponse((b'yes',))
        else:
            return SuccessfulSmartServerResponse((b'no',))