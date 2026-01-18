from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _only_once(f: Callable) -> Callable:
    """decorator to only run a function once"""
    f.called = False

    def wrapped(**kwargs: Any) -> Any:
        if f.called:
            return
        ret = f(**kwargs)
        f.called = True
        return ret
    return wrapped