import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBzrDirRequestDestroyRepository(SmartServerRequestBzrDir):

    def do_bzrdir_request(self, name=None):
        """Destroy the repository.

        New in 2.5.0.

        :return: On success, 'ok'.
        """
        try:
            self._bzrdir.destroy_repository()
        except errors.NoRepositoryPresent as e:
            return FailedSmartServerResponse((b'norepository',))
        return SuccessfulSmartServerResponse((b'ok',))