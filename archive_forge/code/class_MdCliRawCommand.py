from ncclient.operations import OperationError
from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
class MdCliRawCommand(RPC):

    def request(self, command=None):
        node, raw_cmd_node = global_operations('md-cli-raw-command')
        sub_ele(raw_cmd_node, 'md-cli-input-line').text = command
        self._huge_tree = True
        return self._request(node)