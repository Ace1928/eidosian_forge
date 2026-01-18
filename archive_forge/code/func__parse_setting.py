import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
@classmethod
def _parse_setting(cls, name: str):
    parts = name.split('.')
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    else:
        return (parts[0], None, parts[1])