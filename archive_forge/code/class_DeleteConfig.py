from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations import util
from .errors import OperationError
import logging
class DeleteConfig(RPC):
    """`delete-config` RPC"""

    def request(self, target):
        """Delete a configuration datastore.

        *target* specifies the  name or URL of configuration datastore to delete

        :seealso: :ref:`srctarget_params`"""
        node = new_ele('delete-config')
        node.append(util.datastore_or_url('target', target, self._assert))
        return self._request(node)