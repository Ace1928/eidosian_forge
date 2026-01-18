import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class NativeQuilResponse(Message):
    """
    Native Quil and associated metadata returned from quilc.
    """
    quil: str
    'Native Quil returned from quilc.'
    metadata: Optional[NativeQuilMetadata] = None
    'Metadata for the returned Native Quil.'