from ncclient.operations.errors import OperationError
from ncclient.operations.rpc import RPC, RPCReply
from ncclient.xml_ import *
from lxml import etree
from ncclient.operations import util
def _validate_with_defaults_mode(mode, capabilities):
    valid_modes = _get_valid_with_defaults_modes(capabilities)
    if mode.strip().lower() not in valid_modes:
        raise WithDefaultsError("Invalid 'with-defaults' mode '{provided}'; the server only supports the following: {options}".format(provided=mode, options=', '.join(valid_modes)))