import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def _check_section_name(name: bytes) -> bool:
    for i in range(len(name)):
        c = name[i:i + 1]
        if not c.isalnum() and c not in (b'-', b'.'):
            return False
    return True