from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations import util
from .errors import OperationError
import logging
class DiscardChanges(RPC):
    """`discard-changes` RPC. Depends on the `:candidate` capability."""
    DEPENDS = [':candidate']

    def request(self):
        """Revert the candidate configuration to the currently running configuration. Any uncommitted changes are discarded."""
        return self._request(new_ele('discard-changes'))