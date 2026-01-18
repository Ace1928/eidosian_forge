import os
import re
import typing
from typing import Literal, Optional, Tuple
def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)