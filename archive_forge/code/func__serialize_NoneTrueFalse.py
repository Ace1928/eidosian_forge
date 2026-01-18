import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
def _serialize_NoneTrueFalse(self, arg):
    if arg is False:
        return b'False'
    if not arg:
        return b''
    return b'True'