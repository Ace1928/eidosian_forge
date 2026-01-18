from ncclient.operations.errors import OperationError
from ncclient.operations.rpc import RPC, RPCReply
from ncclient.xml_ import *
from lxml import etree
from ncclient.operations import util
def _append_with_defaults_mode(node, mode, capabilities):
    _validate_with_defaults_mode(mode, capabilities)
    with_defaults_element = sub_ele_ns(node, 'with-defaults', NETCONF_WITH_DEFAULTS_NS)
    with_defaults_element.text = mode