import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBzrDirRequestConfigFile(SmartServerRequestBzrDir):

    def do_bzrdir_request(self):
        """Get the configuration bytes for a config file in bzrdir.

        The body is not utf8 decoded - it is the literal bytestream from disk.
        """
        config = self._bzrdir._get_config()
        if config is None:
            content = b''
        else:
            content = config._get_config_file().read()
        return SuccessfulSmartServerResponse((), content)