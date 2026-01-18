from ncclient.xml_ import *
from ncclient.operations import util
from ncclient.operations.rpc import RPC
class GetBulk(RPC):
    """The *get-bulk* RPC."""

    def request(self, filter=None):
        """Retrieve running configuration and device state information.

        *filter* specifies the portion of the configuration to retrieve (by default entire configuration is retrieved)

        :seealso: :ref:`filter_params`
        """
        node = new_ele('get-bulk')
        if filter is not None:
            node.append(util.build_filter(filter))
        return self._request(node)