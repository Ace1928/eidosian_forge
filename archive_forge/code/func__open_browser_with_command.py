from __future__ import annotations
import dataclasses
import functools
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Final, Iterable, Mapping, TypeVar
from streamlit import env_util
def _open_browser_with_command(command: str, url: str) -> None:
    cmd_line = [command, url]
    with open(os.devnull, 'w') as devnull:
        subprocess.Popen(cmd_line, stdout=devnull, stderr=subprocess.STDOUT)