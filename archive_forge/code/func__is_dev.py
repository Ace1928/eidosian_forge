from __future__ import annotations
from collections import namedtuple
import re
def _is_dev(self) -> bool:
    """Is development."""
    return bool(self.release < 'alpha')