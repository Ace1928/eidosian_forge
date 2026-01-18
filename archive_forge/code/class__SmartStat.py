imported from breezy.bzr.smart.
from io import BytesIO
from .. import config, debug, errors, trace, transport, urlutils
from ..bzr import remote
from ..bzr.smart import client, medium
class _SmartStat:

    def __init__(self, size, mode):
        self.st_size = size
        self.st_mode = mode