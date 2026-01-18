from ncclient.xml_ import *
from ncclient.operations.rpc import RaiseMode, RPC
class Unlock(RPC):
    """`unlock` RPC"""

    def request(self, target='candidate'):
        """Release a configuration lock, previously obtained with the lock operation.

        *target* is the name of the configuration datastore to unlock
        """
        node = new_ele('unlock')
        sub_ele(sub_ele(node, 'target'), target)
        return self._request(node)