from ncclient.operations.errors import OperationError
from ncclient.operations.rpc import RPC, RPCReply
from ncclient.xml_ import *
from lxml import etree
from ncclient.operations import util
class GetSchemaReply(GetReply):
    """Reply for GetSchema called with specific parsing hook."""

    def _parsing_hook(self, root):
        self._data = None
        if not self._errors:
            self._data = root.find(qualify('data', NETCONF_MONITORING_NS)).text