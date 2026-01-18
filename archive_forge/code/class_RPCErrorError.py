import uuid
import warnings
from inspect import signature
from typing import Callable, Optional, Tuple, Union, List, Any
import rpcq.messages
class RPCErrorError(IOError):
    """JSON RPC error that is raised by a Client when it receives an RPCError message"""

    def __init__(self, *args, **kwargs):
        if type(self) is RPCErrorError:
            warnings.warn('`RPCErrorError` is deprecated in favor of the less-loquacious `RPCError`.', DeprecationWarning)
        super().__init__(*args, **kwargs)