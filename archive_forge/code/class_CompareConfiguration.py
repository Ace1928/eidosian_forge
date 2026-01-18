from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations.rpc import RPCReply
from ncclient.operations.rpc import RPCError
from ncclient import NCClientError
import math
class CompareConfiguration(RPC):

    def request(self, rollback=0, format='text'):
        node = new_ele('get-configuration', {'compare': 'rollback', 'format': format, 'rollback': str(rollback)})
        return self._request(node)