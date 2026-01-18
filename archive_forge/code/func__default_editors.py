from __future__ import annotations
import os
from os.path import exists
from os.path import join
from os.path import splitext
from subprocess import check_call
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from .compat import is_posix
from .exc import CommandError
def _default_editors() -> List[str]:
    if is_posix:
        return ['sensible-editor', 'editor', 'nano', 'vim', 'code']
    else:
        return ['code.exe', 'notepad++.exe', 'notepad.exe']