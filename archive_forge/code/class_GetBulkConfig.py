from ncclient.xml_ import *
from ncclient.operations import util
from ncclient.operations.rpc import RPC
class GetBulkConfig(RPC):
    """The *get-bulk-config* RPC."""

    def request(self, source, filter=None):
        """Retrieve all or part of a specified configuration.

        *source* name of the configuration datastore being queried

        *filter* specifies the portion of the configuration to retrieve (by default entire configuration is retrieved)

        :seealso: :ref:`filter_params`"""
        node = new_ele('get-bulk-config')
        node.append(util.datastore_or_url('source', source, self._assert))
        if filter is not None:
            node.append(util.build_filter(filter))
        return self._request(node)