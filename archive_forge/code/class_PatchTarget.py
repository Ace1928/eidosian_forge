import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
@dataclass(eq=False, repr=False)
class PatchTarget(Message):
    """
    Patchable memory location descriptor.
    """
    patch_type: ParameterSpec
    'Data type at this address.'
    patch_offset: int
    'Memory address of the patch.'