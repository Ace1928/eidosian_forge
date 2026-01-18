from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations.rpc import RPCReply
from ncclient.operations.rpc import RPCError
from ncclient import NCClientError
import math
class Halt(RPC):

    def request(self):
        node = new_ele('request-halt')
        return self._request(node)