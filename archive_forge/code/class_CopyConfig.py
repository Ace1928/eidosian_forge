from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations import util
from .errors import OperationError
import logging
class CopyConfig(RPC):
    """`copy-config` RPC"""

    def request(self, source, target):
        """Create or replace an entire configuration datastore with the contents of another complete
        configuration datastore.

        *source* is the name of the configuration datastore to use as the source of the copy operation or `config` element containing the configuration subtree to copy

        *target* is the name of the configuration datastore to use as the destination of the copy operation

        :seealso: :ref:`srctarget_params`"""
        node = new_ele('copy-config')
        node.append(util.datastore_or_url('target', target, self._assert))
        try:
            node.append(util.datastore_or_url('source', source, self._assert))
        except Exception:
            node.append(validated_element(source, ('source', qualify('source'))))
        return self._request(node)