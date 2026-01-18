from ncclient.operations.errors import OperationError
from ncclient.operations.rpc import RPC, RPCReply
from ncclient.xml_ import *
from lxml import etree
from ncclient.operations import util
class WithDefaultsError(OperationError):
    """Invalid 'with-defaults' mode or capability URI"""