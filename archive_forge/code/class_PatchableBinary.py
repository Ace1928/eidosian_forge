import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class PatchableBinary(Message):
    """
    Tsunami binary with patching metadata for classical
      parameter modification.
    """
    base_binary: Any
    'Raw Tsunami binary object.'
    patch_table: Dict[str, PatchTarget]
    'Dictionary mapping patch names to their memory\n          descriptors.'