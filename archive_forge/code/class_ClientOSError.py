import asyncio
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
from .http_parser import RawResponseMessage
from .typedefs import LooseHeaders
class ClientOSError(ClientConnectionError, OSError):
    """OSError error."""