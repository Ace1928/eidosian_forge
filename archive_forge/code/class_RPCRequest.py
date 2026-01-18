import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class RPCRequest(Message):
    """
    A single request object according to the JSONRPC standard.
    """
    method: str
    'The RPC function name.'
    params: Any
    'The RPC function arguments.'
    id: str
    'RPC request id (used to verify that request and response belong together).'
    jsonrpc: str = '2.0'
    'The JSONRPC version.'
    client_timeout: Optional[float] = None
    'The client-side timeout for the request. The server itself may be configured with a timeout that is greater than the client-side timeout, in which case the server can choose to terminate any processing of the request.'
    client_key: Optional[str] = None
    'The ZeroMQ CURVE public key used to make the request, as received by the server. Empty if no key is used.'