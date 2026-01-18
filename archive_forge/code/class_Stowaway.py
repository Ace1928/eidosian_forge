from __future__ import annotations
import multiprocessing
import multiprocessing.process
import os
import os.path
import sys
import traceback
from typing import Any
from coverage.debug import DebugControl
class Stowaway:
    """An object to pickle, so when it is unpickled, it can apply the monkey-patch."""

    def __init__(self, rcfile: str) -> None:
        self.rcfile = rcfile

    def __getstate__(self) -> dict[str, str]:
        return {'rcfile': self.rcfile}

    def __setstate__(self, state: dict[str, str]) -> None:
        patch_multiprocessing(state['rcfile'])